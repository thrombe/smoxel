use bevy::{
    core_pipeline::prepass::{DepthPrepass, MotionVectorPrepass, NormalPrepass},
    diagnostic::{DiagnosticsPlugin, FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin},
    input::mouse::MouseMotion,
    pbr::{wireframe::WireframePlugin, NotShadowCaster, PbrPlugin},
    prelude::*,
    reflect::TypePath,
    render::{
        render_resource::{AsBindGroup, ShaderRef, ShaderType},
        settings::PowerPreference,
        RenderPlugin,
    },
    window::{CursorGrabMode, PrimaryWindow},
};

#[cfg(feature = "dev")]
use bevy_inspector_egui::{
    bevy_egui::{self, EguiContexts},
    quick::WorldInspectorPlugin,
};
use chunk::ChunkMaterial;

fn main() {
    let mut app = App::new();

    app.add_plugins((
        DefaultPlugins
            .set(RenderPlugin {
                render_creation: bevy::render::settings::RenderCreation::Automatic(
                    bevy::render::settings::WgpuSettings {
                        // power_preference: PowerPreference::LowPower,
                        ..Default::default()
                    },
                ),
            })
            .set(WindowPlugin {
                primary_window: Some(Window {
                    title: "Smoxel".into(),
                    present_mode: bevy::window::PresentMode::Fifo,
                    // present_mode: bevy::window::PresentMode::Immediate,
                    // present_mode: bevy::window::PresentMode::Mailbox,
                    ..Default::default()
                }),
                ..Default::default()
            })
            .set(PbrPlugin {
                // The prepass is enabled by default on the StandardMaterial,
                // but you can disable it if you need to.
                //
                // prepass_enabled: false,
                ..default()
            })
            .set(AssetPlugin {
                watch_for_changes_override: Some(true),
                ..Default::default()
            }),
        MaterialPlugin::<CustomMaterial>::default(),
        // MaterialPlugin::<ChunkMaterial>::default(),
        // MaterialPlugin::<PrepassOutputMaterial> {
        //     // This material only needs to read the prepass textures,
        //     // but the meshes using it should not contribute to the prepass render, so we can disable it.
        //     prepass_enabled: false,
        //     ..default()
        // },
    ))
    .add_state::<AppState>()
    .add_state::<ControlsState>()
    .add_plugins((
        // DiagnosticsPlugin,
        FrameTimeDiagnosticsPlugin,
        LogDiagnosticsPlugin::filtered(vec![
            FrameTimeDiagnosticsPlugin::FPS,
            FrameTimeDiagnosticsPlugin::FRAME_TIME,
        ]),
    ))
    .add_plugins(WireframePlugin)
    .add_plugins(render::RenderPlugin)
    .add_plugins(player::PlayerPlugin)
    .add_plugins(spectator::Spectator)
    .add_plugins(chunk::VoxelPlugin)
    .add_plugins(vox::VoxLoader)
    .add_systems(Startup, setup)
    .add_systems(
        Update,
        (
            rotate,
            // toggle_prepass_view
        ),
    )
    // Disabling MSAA for maximum compatibility. Shader prepass with MSAA needs GPU capability MULTISAMPLED_SHADING
    .insert_resource(Msaa::Off);

    #[cfg(feature = "dev")]
    app.add_systems(
        PreUpdate,
        (absorb_egui_inputs,)
            .after(bevy_egui::systems::process_input_system)
            .after(bevy_egui::EguiSet::ProcessInput), // .after(bevy_egui::EguiSet::ProcessOutput), // .before(bevy_egui::EguiSet::BeginFrame),
    )
    .add_plugins(WorldInspectorPlugin::new());

    app.run();
}

#[cfg(feature = "dev")]
fn absorb_egui_inputs(mut mouse: ResMut<Input<MouseButton>>, mut contexts: EguiContexts) {
    let ctx = contexts.ctx_mut();
    // ctx.wants_pointer_input(); // NOTE: this egui method is broken. it returns false on the frame the thing is clicked
    // dbg!(
    //     ctx.wants_pointer_input(),
    //     ctx.is_pointer_over_area(),
    //     ctx.is_using_pointer(),
    //     mouse.any_just_pressed([MouseButton::Left, MouseButton::Right, MouseButton::Middle]),
    //     mouse.any_pressed([MouseButton::Left, MouseButton::Right, MouseButton::Middle])
    // );
    if ctx.is_using_pointer()
        || ctx.is_pointer_over_area()
            && mouse.any_just_pressed([MouseButton::Left, MouseButton::Right, MouseButton::Middle])
    {
        mouse.reset_all();
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Default, States)]
enum AppState {
    #[default]
    Loading,
    Playing,
}
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Default, States)]
enum ControlsState {
    #[default]
    Player,
    Spectator,
}

mod render {
    use std::{cmp::Reverse, ops::Range};

    use bevy::{
        app::{Plugin, Startup},
        asset::{load_internal_asset, AssetApp, AssetEvent, AssetId, AssetServer, Assets, Handle},
        core::Name,
        core_pipeline::{
            clear_color::ClearColorConfig,
            core_3d::{
                self,
                graph::{input::VIEW_ENTITY, node::{MAIN_OPAQUE_PASS, START_MAIN_PASS, END_MAIN_PASS}},
                AlphaMask3d, Camera3d, Camera3dBundle, Opaque3d, ScreenSpaceTransmissionQuality,
                Transmissive3d, Transparent3d, CORE_3D,
            },
            prepass::{DeferredPrepass, DepthPrepass, MotionVectorPrepass, NormalPrepass},
            tonemapping::{DebandDither, Tonemapping},
        },
        ecs::{
            bundle::Bundle,
            component::Component,
            entity::Entity,
            event::EventReader,
            query::{Has, QueryItem, QueryState, ROQueryItem, With},
            schedule::{IntoSystemConfigs, NextState, OnEnter, State, States},
            system::{
                lifetimeless::{Read, SQuery, SRes},
                Commands, Local, Query, ReadOnlySystemParam, Res, ResMut, Resource, RunSystemOnce,
                SystemParamItem, SystemState,
            },
            world::{FromWorld, World},
        },
        math::{UVec2, Vec2, Vec3},
        pbr::{
            AlphaMode, DrawMesh, EnvironmentMapLight, Material, MaterialBindGroupId,
            MaterialMeshBundle, MaterialProperties, MeshFlags, MeshPipeline, MeshPipelineKey,
            MeshPipelineViewLayoutKey, MeshTransforms, OpaqueRendererMethod, PreparedMaterial,
            RenderMaterialInstances, RenderMaterials, RenderMeshInstance, RenderMeshInstances,
            ScreenSpaceAmbientOcclusionSettings, SetMaterialBindGroup, SetMeshBindGroup,
            SetMeshViewBindGroup, ShadowFilteringMethod, PrepassPipelinePlugin, Mesh3d, MeshUniform,
        },
        reflect::Reflect,
        render::{
            camera::{
                Camera, CameraRenderGraph, OrthographicProjection, Projection, RenderTarget,
                ScalingMode, TemporalJitter, ExtractedCamera,
            },
            extract_component::{ExtractComponent, ExtractComponentPlugin, ComponentUniforms},
            extract_instances::ExtractInstancesPlugin,
            extract_resource::{ExtractResource, ExtractResourcePlugin},
            main_graph::node::CAMERA_DRIVER,
            mesh::{
                shape::{Cube, UVSphere},
                GpuBufferInfo, Mesh, MeshVertexBufferLayout,
            },
            primitives::{Frustum, Sphere},
            render_asset::{prepare_assets, RenderAssets},
            render_graph::{
                NodeRunError, RenderGraph, RenderGraphApp, RenderGraphContext, SlotInfo, SlotType,
                ViewNode, ViewNodeRunner,
            },
            render_phase::{
                AddRenderCommand, CachedRenderPipelinePhaseItem, Draw, DrawFunctionId,
                DrawFunctions, PhaseItem, RenderCommand, RenderCommandResult, RenderPhase,
                SetItemPipeline, TrackedRenderPass, sort_phase_system,
            },
            render_resource::{
                AsBindGroup, AsBindGroupError, BindGroup, BindGroupEntry, BindGroupLayout,
                BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingResource,
                BindingType::{self, StorageTexture},
                BufferBindingType, BufferSize, CachedRenderPipelineId, Extent3d, Operations,
                PipelineCache, PrimitiveTopology, RenderPassDepthStencilAttachment,
                RenderPassDescriptor, RenderPipelineDescriptor, Shader, ShaderSize, ShaderStages,
                ShaderType, SpecializedMeshPipeline, SpecializedMeshPipelineError,
                SpecializedMeshPipelines, StorageTextureAccess, Texture, TextureDescriptor,
                TextureDimension, TextureFormat, TextureUsages, TextureView, TextureViewDescriptor,
                TextureViewDimension, UniformBuffer, LoadOp, BindGroupEntries, Face, TextureAspect, ShaderDefVal, ColorTargetState, ColorWrites, TextureSampleType,
            },
            renderer::{RenderContext, RenderDevice, RenderQueue},
            texture::{FallbackImage, Image},
            view::{
                ExtractedView, InheritedVisibility, Msaa, NoFrustumCulling, ViewTarget,
                ViewVisibility, Visibility, VisibleEntities, ViewDepthTexture,
            },
            Extract, ExtractSchedule, Render, RenderApp, RenderSet, batching::{batch_and_prepare_render_phase, GetBatchData, write_batched_instance_buffer}, color::Color,
        },
        transform::components::{GlobalTransform, Transform},
        utils::{nonmax::NonMaxU32, FloatOrd, HashSet},
        window::Window,
    };

    use crate::{
        chunk::ChunkMaterial,
        player::{PlayerEntity, PlayerPlugin},
    };

    pub struct RenderPlugin;

    impl Plugin for RenderPlugin {
        fn build(&self, app: &mut bevy::prelude::App) {
            app.add_plugins(WorldPlugin)
                .add_plugins(ChunkMaterialPlugin);
        }

        fn finish(&self, app: &mut bevy::prelude::App) {}
    }

    struct WorldPlugin;
    impl Plugin for WorldPlugin {
        fn build(&self, app: &mut bevy::prelude::App) {}

        fn finish(&self, app: &mut bevy::prelude::App) {
            let render_device = app.world.resource::<RenderDevice>();
            let render_queue = app.world.resource::<RenderQueue>();

            let transient_world_size = Extent3d {
                width: 100,
                height: 100,
                depth_or_array_layers: 100,
            };
            let transient_world_texture = render_device.create_texture_with_data(
                render_queue,
                &TextureDescriptor {
                    label: Some("transient world texture"),
                    size: transient_world_size,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: TextureDimension::D3,
                    format: TextureFormat::R8Uint,
                    usage: TextureUsages::STORAGE_BINDING | TextureUsages::COPY_DST,
                    view_formats: &[],
                },
                &[0; 100usize.pow(3)],
            );
            let transient_world_texture_view =
                transient_world_texture.create_view(&TextureViewDescriptor::default());

            let depth_prepass_texture = render_device.create_texture(
                &TextureDescriptor {
                    label: Some("depth prepass texture"),
                    size: Extent3d {
                        width: 640,
                        height: 360,
                        // width: 853,
                        // height: 480,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: TextureDimension::D2,
                    format: TextureFormat::R32Float,
                    usage: TextureUsages::TEXTURE_BINDING | TextureUsages::RENDER_ATTACHMENT,
                    view_formats: &[],
                }
            );
            let depth_prepass_texture_view =
                depth_prepass_texture.create_view(&TextureViewDescriptor {
                    // aspect: TextureAspect::StencilOnly,
                    ..Default::default()
                });

            // TODO: can use AsBindGroup to generate this boilerplate
            let render_pass_bind_group_layout =
                render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: Some("world data bind group"),
                    entries: &[
                        BindGroupLayoutEntry {
                            binding: 0,
                            visibility: ShaderStages::FRAGMENT | ShaderStages::COMPUTE,
                            ty: BindingType::Buffer {
                                ty: BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: BufferSize::new(
                                    WorldDataUniforms::SHADER_SIZE.into(),
                                ),
                            },
                            count: None,
                        },
                        BindGroupLayoutEntry {
                            binding: 1,
                            visibility: ShaderStages::FRAGMENT | ShaderStages::COMPUTE,
                            ty: BindingType::StorageTexture {
                                access: StorageTextureAccess::ReadWrite,
                                format: TextureFormat::R8Uint,
                                view_dimension: TextureViewDimension::D3,
                            },
                            count: None,
                        },
                        BindGroupLayoutEntry {
                            binding: 2,
                            visibility: ShaderStages::FRAGMENT | ShaderStages::COMPUTE,
                            ty: BindingType::Texture {
                                view_dimension: TextureViewDimension::D2,
                                sample_type: TextureSampleType::Float { filterable: false },
                                multisampled: false,
                            },
                            count: None,
                        },
                    ],
                });
            let depth_pass_bind_group_layout =
                render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: Some("depth pass bind group"),
                    entries: &[
                        BindGroupLayoutEntry {
                            binding: 0,
                            visibility: ShaderStages::FRAGMENT | ShaderStages::COMPUTE,
                            ty: BindingType::Buffer {
                                ty: BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: BufferSize::new(
                                    WorldDataUniforms::SHADER_SIZE.into(),
                                ),
                            },
                            count: None,
                        },
                        BindGroupLayoutEntry {
                            binding: 1,
                            visibility: ShaderStages::FRAGMENT | ShaderStages::COMPUTE,
                            ty: BindingType::StorageTexture {
                                access: StorageTextureAccess::ReadWrite,
                                format: TextureFormat::R8Uint,
                                view_dimension: TextureViewDimension::D3,
                            },
                            count: None,
                        },
                    ],
                });

            let mut uniforms = UniformBuffer::from(WorldDataUniforms::default());
            uniforms.write_buffer(render_device, render_queue);

            let render_pass_bind_group = render_device.create_bind_group(
                None,
                &render_pass_bind_group_layout,
                &[
                    BindGroupEntry {
                        binding: 0,
                        resource: uniforms.binding().unwrap(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::TextureView(&transient_world_texture_view),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: BindingResource::TextureView(&depth_prepass_texture_view),
                    },
                ],
            );
            let depth_pass_bind_group = render_device.create_bind_group(
                None,
                &depth_pass_bind_group_layout,
                &[
                    BindGroupEntry {
                        binding: 0,
                        resource: uniforms.binding().unwrap(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::TextureView(&transient_world_texture_view),
                    },
                ],
            );

            // NOTE: i am assuming we need to have this image just so that we can reuse the rendering pipeline for
            // voxelization. and we don't actually need this image's output. the real output will be done directly
            // the 3d world texture
            let transient_world_camera_target_image = Image {
                texture_descriptor: TextureDescriptor {
                    label: Some("transient world camera target"),
                    size: Extent3d {
                        depth_or_array_layers: 1,
                        ..transient_world_size
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: TextureDimension::D2,
                    format: TextureFormat::R8Unorm,
                    usage: TextureUsages::TEXTURE_BINDING | TextureUsages::RENDER_ATTACHMENT,
                    view_formats: &[TextureFormat::R8Unorm],
                },
                data: vec![0; 100usize.pow(2)],
                ..Default::default()
            };

            let depth_prepass_depth_stencil = render_device.create_texture(
                &TextureDescriptor {
                    label: Some("depth prepass depth stencil"),
                    size: Extent3d {
                        width: 640,
                        height: 360,
                        // width: 853,
                        // height: 480,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: TextureDimension::D2,
                    format: TextureFormat::Depth32Float,
                    usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                },
            );
            let depth_prepass_depth_stencil_view =
                depth_prepass_depth_stencil.create_view(&TextureViewDescriptor {
                    // label: Some("depth prepass depth stencil view"),
                    // format: Some(TextureFormat::Depth32Float),
                    // dimension: None,
                    // aspect: TextureAspect::DepthOnly,
                    ..Default::default()
                });

            let mut images = app.world.resource_mut::<Assets<Image>>();
            let transient_world_camera_target = images.add(transient_world_camera_target_image);

            let render_app = app.sub_app_mut(RenderApp);
            render_app.insert_resource(WorldData {
                render_pass_bind_group_layout,
                render_pass_bind_group,
                depth_pass_bind_group_layout,
                depth_pass_bind_group,
                transient_world_texture,
                transient_world_texture_view,
                transient_world_camera_target,
                depth_prepass_texture,
                depth_prepass_texture_view,
                depth_prepass_depth_stencil,
                depth_prepass_depth_stencil_view,
                uniforms,
            });
            render_app.init_resource::<WorldDataUniforms>();
        }
    }

    #[derive(Resource)]
    struct WorldData {
        render_pass_bind_group_layout: BindGroupLayout,
        render_pass_bind_group: BindGroup,
        depth_pass_bind_group_layout: BindGroupLayout,
        depth_pass_bind_group: BindGroup,

        transient_world_texture: Texture,
        transient_world_texture_view: TextureView,
        transient_world_camera_target: Handle<Image>,

        depth_prepass_texture: Texture,
        depth_prepass_texture_view: TextureView,
        depth_prepass_depth_stencil: Texture,
        depth_prepass_depth_stencil_view: TextureView,

        uniforms: UniformBuffer<WorldDataUniforms>,
    }

    #[derive(Default, Debug, Clone, ShaderType, Resource)]
    struct WorldDataUniforms {
        player_pos: Vec3,
        screen_resolution: UVec2,
    }

    // refer the source code of MaterialPlugin<M>
    struct ChunkMaterialPlugin;
    impl Plugin for ChunkMaterialPlugin {
        fn build(&self, app: &mut bevy::prelude::App) {
            app.init_asset::<ChunkMaterial>()
                .add_plugins(ExtractInstancesPlugin::<AssetId<ChunkMaterial>>::extract_visible());

            let Ok(render_app) = app.get_sub_app_mut(RenderApp) else {
                // TODO: figure out why doing this is fine.
                // i would assume that not adding the systems below this if the renderer is not initialized *yet*
                // will cause problems. but bevy code does this too.
                return;
            };

            render_app
                // Add the node that will render the custom phase
                .add_render_graph_node::<ViewNodeRunner<VoxelRenderNode>>(
                    CORE_3D,
                    VoxelRenderNode::NAME,
                )
                // This will schedule the custom node to run after the main opaque pass
                .add_render_graph_edge(CORE_3D, MAIN_OPAQUE_PASS, VoxelRenderNode::NAME);

            render_app
                .add_systems(ExtractSchedule, extract_render_phase)
                .add_systems(ExtractSchedule, extract_world_data)
                .init_resource::<DrawFunctions<ChunkDepthPhaseItem>>()
                .init_resource::<DrawFunctions<ChunkRenderPhaseItem>>()
                .add_render_command::<ChunkDepthPhaseItem, ChunkDepthPrepass<ChunkMaterial>>()
                .add_render_command::<ChunkRenderPhaseItem, DrawMaterial<ChunkMaterial>>()
                .init_resource::<ExtractedMaterials<ChunkMaterial>>()
                .init_resource::<RenderMaterials<ChunkMaterial>>()
                .init_resource::<SpecializedMeshPipelines<ChunkPipeline>>()
                .init_resource::<SpecializedMeshPipelines<ChunkDepthPipeline>>()
                .add_systems(ExtractSchedule, extract_materials::<ChunkMaterial>)
                .add_systems(
                    Render,
                    (
                        prepare_materials::<ChunkMaterial>
                            .in_set(RenderSet::PrepareAssets)
                            .after(prepare_assets::<Image>),
                        sort_phase_system::<ChunkDepthPhaseItem>.in_set(RenderSet::PhaseSort),
                        sort_phase_system::<ChunkRenderPhaseItem>.in_set(RenderSet::PhaseSort),
                        queue_material_meshes::<ChunkMaterial>
                            .in_set(RenderSet::QueueMeshes)
                            .after(prepare_materials::<ChunkMaterial>),
                        (
                            batch_and_prepare_render_phase::<ChunkRenderPhaseItem, MeshPipeline>,
                            batch_and_prepare_render_phase::<ChunkDepthPhaseItem, MeshPipeline>,
                        ).in_set(RenderSet::PrepareResources),
                    ),
                );
        }
        fn finish(&self, app: &mut bevy::prelude::App) {
            let render_app = app.sub_app_mut(RenderApp);

            render_app.init_resource::<ChunkPipeline>();
            render_app.init_resource::<ChunkDepthPipeline>();

            render_app.add_systems(Render, update_world_data.in_set(RenderSet::Prepare));
        }
    }

    fn extract_render_phase(
        mut commands: Commands,
        cameras_3d: Extract<Query<(Entity, &Camera), With<Camera3d>>>,
    ) {
        for (entity, camera) in &cameras_3d {
            if camera.is_active {
                commands
                    .get_or_spawn(entity)
                    .insert(RenderPhase::<ChunkDepthPhaseItem>::default())
                    .insert(RenderPhase::<ChunkRenderPhaseItem>::default());
            }
        }
    }

    #[derive(Default)]
    struct VoxelRenderNode;
    impl VoxelRenderNode {
        const NAME: &'static str = "voxel_render_node";
    }
    impl ViewNode for VoxelRenderNode {
        type ViewQuery = (
            &'static RenderPhase<ChunkDepthPhaseItem>,
            &'static RenderPhase<ChunkRenderPhaseItem>,
            &'static ViewTarget,
            &'static ViewDepthTexture,
            &'static Camera3d,
            &'static ExtractedCamera,
        );

        fn run(
            &self,
            graph: &mut RenderGraphContext,
            render_context: &mut RenderContext,
            (depth_phase, render_phase, target, bevy_depth_view, cam3d, cam): QueryItem<Self::ViewQuery>,
            world: &bevy::prelude::World,
        ) -> Result<(), NodeRunError> {
            if render_phase.items.is_empty() {
                return Ok(());
            }

            let world_data = world.resource::<WorldData>();

            {
                let mut render_pass =
                    render_context.begin_tracked_render_pass(RenderPassDescriptor {
                        label: Some("chunk_depth_prepass"),
                        // color_attachments: &[Some(target.get_color_attachment(Operations {
                        //     load: LoadOp::Load,
                        //     store: true,
                        // }))],
                        color_attachments: &[Some(
                            bevy::render::render_resource::RenderPassColorAttachment {
                                view: &world_data.depth_prepass_texture_view,
                                resolve_target: None,
                                ops: Operations {
                                    load: LoadOp::Clear(Color::rgb_u8(0, 0, 0).into()),
                                    store: true,
                                },
                            },
                        )],
                        // depth_stencil_attachment: None,
                        depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                            view: &world_data.depth_prepass_depth_stencil_view,
                            depth_ops: Some(Operations {
                                // load: LoadOp::Load,
                                load: LoadOp::Clear(Default::default()),
                                store: true,
                            }),
                            stencil_ops: None,
                        }),
                    });

                if let Some(viewport) = cam.viewport.as_ref() {
                    render_pass.set_camera_viewport(viewport);
                }

                // This will automatically call the draw command on each items in the render phase
                depth_phase.render(&mut render_pass, world, graph.view_entity());
            }

            {
                let mut render_pass =
                    render_context.begin_tracked_render_pass(RenderPassDescriptor {
                        label: Some("chunk_render_pass"),
                        color_attachments: &[Some(target.get_color_attachment(Operations {
                            load: LoadOp::Load,
                            store: true,
                        }))],
                        depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                            view: &bevy_depth_view.view,
                            depth_ops: Some(Operations {
                                // load: cam3d.depth_load_op.clone().into(),
                                load: LoadOp::Load,
                                store: true,
                            }),
                            stencil_ops: None,
                        }),
                    });

                if let Some(viewport) = cam.viewport.as_ref() {
                    render_pass.set_camera_viewport(viewport);
                }

                // This will automatically call the draw command on each items in the render phase
                render_phase.render(&mut render_pass, world, graph.view_entity());
            }

            Ok(())
        }
    }

    pub struct ChunkRenderPhaseItem {
        pub distance: f32,
        pub pipeline: CachedRenderPipelineId,
        pub entity: Entity,
        pub draw_function: DrawFunctionId,
        pub batch_range: Range<u32>,
        pub dynamic_offset: Option<NonMaxU32>,
    }
    impl PhaseItem for ChunkRenderPhaseItem {
        // NOTE: Values increase towards the camera. Front-to-back ordering for opaque means we need a descending sort.
        type SortKey = Reverse<FloatOrd>;

        #[inline]
        fn entity(&self) -> Entity {
            self.entity
        }

        #[inline]
        fn sort_key(&self) -> Self::SortKey {
            Reverse(FloatOrd(self.distance))
        }

        #[inline]
        fn draw_function(&self) -> DrawFunctionId {
            self.draw_function
        }

        #[inline]
        fn batch_range(&self) -> &Range<u32> {
            &self.batch_range
        }

        #[inline]
        fn batch_range_mut(&mut self) -> &mut Range<u32> {
            &mut self.batch_range
        }

        #[inline]
        fn dynamic_offset(&self) -> Option<NonMaxU32> {
            self.dynamic_offset
        }

        #[inline]
        fn dynamic_offset_mut(&mut self) -> &mut Option<NonMaxU32> {
            &mut self.dynamic_offset
        }
    }
    impl CachedRenderPipelinePhaseItem for ChunkRenderPhaseItem {
        #[inline]
        fn cached_pipeline(&self) -> CachedRenderPipelineId {
            self.pipeline
        }
    }
    pub struct ChunkDepthPhaseItem {
        pub distance: f32,
        pub pipeline: CachedRenderPipelineId,
        pub entity: Entity,
        pub draw_function: DrawFunctionId,
        pub batch_range: Range<u32>,
        pub dynamic_offset: Option<NonMaxU32>,
    }
    impl PhaseItem for ChunkDepthPhaseItem {
        // NOTE: Values increase towards the camera. Front-to-back ordering for opaque means we need a descending sort.
        type SortKey = Reverse<FloatOrd>;

        #[inline]
        fn entity(&self) -> Entity {
            self.entity
        }

        #[inline]
        fn sort_key(&self) -> Self::SortKey {
            Reverse(FloatOrd(self.distance))
        }

        #[inline]
        fn draw_function(&self) -> DrawFunctionId {
            self.draw_function
        }

        #[inline]
        fn batch_range(&self) -> &Range<u32> {
            &self.batch_range
        }

        #[inline]
        fn batch_range_mut(&mut self) -> &mut Range<u32> {
            &mut self.batch_range
        }

        #[inline]
        fn dynamic_offset(&self) -> Option<NonMaxU32> {
            self.dynamic_offset
        }

        #[inline]
        fn dynamic_offset_mut(&mut self) -> &mut Option<NonMaxU32> {
            &mut self.dynamic_offset
        }
    }
    impl CachedRenderPipelinePhaseItem for ChunkDepthPhaseItem {
        #[inline]
        fn cached_pipeline(&self) -> CachedRenderPipelineId {
            self.pipeline
        }
    }

    fn extract_world_data(
        mut commands: Commands,
        player: Extract<Query<&GlobalTransform, With<PlayerEntity>>>,
        windows: Extract<Query<&Window>>,
    ) {
        let Ok(window) = windows.get_single() else {
            return;
        };
        let player = player.single();
        let width = window.resolution.physical_width() as _;
        let height = window.resolution.physical_height() as _;

        commands.insert_resource(WorldDataUniforms {
            player_pos: player.translation(),
            screen_resolution: UVec2::new(width, height),
        });
    }

    fn update_world_data(
        mut world_data: ResMut<WorldData>,
        world_data_uniforms: Res<WorldDataUniforms>,
        render_device: Res<RenderDevice>,
        render_queue: Res<RenderQueue>,
    ) {
        world_data.uniforms.set(world_data_uniforms.clone());
        world_data
            .uniforms
            .write_buffer(&render_device, &render_queue);

        let bind_group = render_device.create_bind_group(
            Some("world data bind group"),
            &world_data.render_pass_bind_group_layout,
            &[
                BindGroupEntry {
                    binding: 0,
                    resource: world_data.uniforms.binding().unwrap(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(
                        &world_data.transient_world_texture_view,
                    ),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(&world_data.depth_prepass_texture_view),
                },
            ],
        );
        world_data.render_pass_bind_group = bind_group;

        let bind_group = render_device.create_bind_group(
            Some("world data bind group"),
            &world_data.depth_pass_bind_group_layout,
            &[
                BindGroupEntry {
                    binding: 0,
                    resource: world_data.uniforms.binding().unwrap(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(
                        &world_data.transient_world_texture_view,
                    ),
                },
            ],
        );
        world_data.depth_pass_bind_group = bind_group;
    }

    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub fn queue_material_meshes<M: Material>(
        depth_draw_functions: Res<DrawFunctions<ChunkDepthPhaseItem>>,
        render_draw_functions: Res<DrawFunctions<ChunkRenderPhaseItem>>,
        material_pipeline: Res<ChunkPipeline>,
        chunk_depth_pipeline: Res<ChunkDepthPipeline>,
        mut pipelines: ResMut<SpecializedMeshPipelines<ChunkPipeline>>,
        mut chunk_depth_pipelines: ResMut<SpecializedMeshPipelines<ChunkDepthPipeline>>,
        pipeline_cache: Res<PipelineCache>,
        msaa: Res<Msaa>,
        render_meshes: Res<RenderAssets<Mesh>>,
        render_materials: Res<RenderMaterials<M>>,
        mut render_mesh_instances: ResMut<RenderMeshInstances>,
        render_material_instances: Res<RenderMaterialInstances<M>>,
        mut views: Query<(
            &ExtractedView,
            &VisibleEntities,
            Option<&Tonemapping>,
            (Has<NormalPrepass>, Has<DepthPrepass>),
            Option<&Camera3d>,
            Option<&TemporalJitter>,
            Option<&Projection>,
            &mut RenderPhase<ChunkDepthPhaseItem>,
            &mut RenderPhase<ChunkRenderPhaseItem>,
        )>,
    ) where
        M::Data: PartialEq + Eq + std::hash::Hash + Clone,
    {
        for (
            view,
            visible_entities,
            tonemapping,
            (normal_prepass, depth_prepass),
            camera_3d,
            temporal_jitter,
            projection,
            mut depth_phase,
            mut render_phase,
        ) in &mut views
        {
            let draw_depth = depth_draw_functions.read().id::<ChunkDepthPrepass<M>>();
            let draw_render = render_draw_functions.read().id::<DrawMaterial<M>>();

            let mut view_key = MeshPipelineKey::from_msaa_samples(msaa.samples())
                | MeshPipelineKey::from_hdr(view.hdr);

            if normal_prepass {
                view_key |= MeshPipelineKey::NORMAL_PREPASS;
            }

            if depth_prepass {
                view_key |= MeshPipelineKey::DEPTH_PREPASS;
            }

            if temporal_jitter.is_some() {
                view_key |= MeshPipelineKey::TEMPORAL_JITTER;
            }

            if let Some(projection) = projection {
                view_key |= match projection {
                    Projection::Perspective(_) => MeshPipelineKey::VIEW_PROJECTION_PERSPECTIVE,
                    Projection::Orthographic(_) => MeshPipelineKey::VIEW_PROJECTION_ORTHOGRAPHIC,
                };
            }

            if !view.hdr {
                if let Some(tonemapping) = tonemapping {
                    view_key |= MeshPipelineKey::TONEMAP_IN_SHADER;
                    view_key |= tonemapping_pipeline_key(*tonemapping);
                }
            }
            let rangefinder = view.rangefinder3d();
            for visible_entity in &visible_entities.entities {
                let Some(material_asset_id) = render_material_instances.get(visible_entity) else {
                    continue;
                };
                let Some(mesh_instance) = render_mesh_instances.get_mut(visible_entity) else {
                    continue;
                };
                let Some(mesh) = render_meshes.get(mesh_instance.mesh_asset_id) else {
                    continue;
                };
                let Some(material) = render_materials.get(material_asset_id) else {
                    continue;
                };

                let mut mesh_key = view_key;

                mesh_key |= MeshPipelineKey::from_primitive_topology(mesh.primitive_topology);

                if mesh.morph_targets.is_some() {
                    mesh_key |= MeshPipelineKey::MORPH_TARGETS;
                }

                let render_pipeline_id = pipelines.specialize(
                    &pipeline_cache,
                    &material_pipeline,
                    mesh_key,
                    &mesh.layout,
                );
                let render_pipeline_id = match render_pipeline_id {
                    Ok(id) => id,
                    Err(err) => {
                        bevy::log::error!("{}", err);
                        continue;
                    }
                };

                let depth_pipeline_id = chunk_depth_pipelines.specialize(
                    &pipeline_cache,
                    &chunk_depth_pipeline,
                    mesh_key,
                    &mesh.layout,
                );
                let depth_pipeline_id = match depth_pipeline_id {
                    Ok(id) => id,
                    Err(err) => {
                        bevy::log::error!("{}", err);
                        continue;
                    }
                };

                mesh_instance.material_bind_group_id = material.get_bind_group_id();

                let distance = rangefinder
                    .distance_translation(&mesh_instance.transforms.transform.translation)
                    + material.properties.depth_bias;

                depth_phase.add(ChunkDepthPhaseItem {
                    entity: *visible_entity,
                    draw_function: draw_depth,
                    pipeline: depth_pipeline_id,
                    distance,
                    batch_range: 0..1,
                    dynamic_offset: None,
                });
                render_phase.add(ChunkRenderPhaseItem {
                    entity: *visible_entity,
                    draw_function: draw_render,
                    pipeline: render_pipeline_id,
                    distance,
                    batch_range: 0..1,
                    dynamic_offset: None,
                });
            }
        }
    }
    const fn tonemapping_pipeline_key(tonemapping: Tonemapping) -> MeshPipelineKey {
        match tonemapping {
            Tonemapping::None => MeshPipelineKey::TONEMAP_METHOD_NONE,
            Tonemapping::Reinhard => MeshPipelineKey::TONEMAP_METHOD_REINHARD,
            Tonemapping::ReinhardLuminance => MeshPipelineKey::TONEMAP_METHOD_REINHARD_LUMINANCE,
            Tonemapping::AcesFitted => MeshPipelineKey::TONEMAP_METHOD_ACES_FITTED,
            Tonemapping::AgX => MeshPipelineKey::TONEMAP_METHOD_AGX,
            Tonemapping::SomewhatBoringDisplayTransform => {
                MeshPipelineKey::TONEMAP_METHOD_SOMEWHAT_BORING_DISPLAY_TRANSFORM
            }
            Tonemapping::TonyMcMapface => MeshPipelineKey::TONEMAP_METHOD_TONY_MC_MAPFACE,
            Tonemapping::BlenderFilmic => MeshPipelineKey::TONEMAP_METHOD_BLENDER_FILMIC,
        }
    }

    /// All [`Material`] values of a given type that should be prepared next frame.
    pub struct PrepareNextFrameMaterials<M: Material> {
        assets: Vec<(AssetId<M>, M)>,
    }

    impl<M: Material> Default for PrepareNextFrameMaterials<M> {
        fn default() -> Self {
            Self {
                assets: Default::default(),
            }
        }
    }
    #[derive(Resource)]
    pub struct ExtractedMaterials<M: Material> {
        extracted: Vec<(AssetId<M>, M)>,
        removed: Vec<AssetId<M>>,
    }
    impl<M: Material> Default for ExtractedMaterials<M> {
        fn default() -> Self {
            Self {
                extracted: Default::default(),
                removed: Default::default(),
            }
        }
    }
    /// This system prepares all assets of the corresponding [`Material`] type
    /// which where extracted this frame for the GPU.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_materials<M: Material>(
        mut prepare_next_frame: Local<PrepareNextFrameMaterials<M>>,
        mut extracted_assets: ResMut<ExtractedMaterials<M>>,
        mut render_materials: ResMut<RenderMaterials<M>>,
        render_device: Res<RenderDevice>,
        images: Res<RenderAssets<Image>>,
        fallback_image: Res<FallbackImage>,
        pipeline: Res<ChunkPipeline>,
        depth_pipeline: Res<ChunkDepthPipeline>,
    ) {
        let queued_assets = std::mem::take(&mut prepare_next_frame.assets);
        for (id, material) in queued_assets.into_iter() {
            match prepare_material(
                &material,
                &render_device,
                &images,
                &fallback_image,
                &pipeline,
            ) {
                Ok(prepared_asset) => {
                    render_materials.insert(id, prepared_asset);
                }
                Err(AsBindGroupError::RetryNextUpdate) => {
                    prepare_next_frame.assets.push((id, material));
                }
            }
        }

        for removed in std::mem::take(&mut extracted_assets.removed) {
            render_materials.remove(&removed);
        }

        for (id, material) in std::mem::take(&mut extracted_assets.extracted) {
            match prepare_material(
                &material,
                &render_device,
                &images,
                &fallback_image,
                &pipeline,
            ) {
                Ok(prepared_asset) => {
                    render_materials.insert(id, prepared_asset);
                }
                Err(AsBindGroupError::RetryNextUpdate) => {
                    prepare_next_frame.assets.push((id, material));
                }
            }
        }
    }

    fn prepare_material<M: Material>(
        material: &M,
        render_device: &RenderDevice,
        images: &RenderAssets<Image>,
        fallback_image: &FallbackImage,
        pipeline: &ChunkPipeline,
    ) -> Result<PreparedMaterial<M>, AsBindGroupError> {
        let prepared = material.as_bind_group(
            &pipeline.material_bind_group_layout,
            render_device,
            images,
            fallback_image,
        )?;
        Ok(PreparedMaterial {
            bindings: prepared.bindings,
            bind_group: prepared.bind_group,
            key: prepared.data,
            properties: MaterialProperties {
                alpha_mode: material.alpha_mode(),
                depth_bias: material.depth_bias(),
                reads_view_transmission_texture: material.reads_view_transmission_texture(),
                render_method: OpaqueRendererMethod::Forward,
            },
        })
    }
    /// This system extracts all created or modified assets of the corresponding [`Material`] type
    /// into the "render world".
    pub fn extract_materials<M: Material>(
        mut commands: Commands,
        mut events: Extract<EventReader<AssetEvent<M>>>,
        assets: Extract<Res<Assets<M>>>,
    ) {
        let mut changed_assets = HashSet::default();
        let mut removed = Vec::new();
        for event in events.read() {
            match event {
                AssetEvent::Added { id } | AssetEvent::Modified { id } => {
                    changed_assets.insert(*id);
                }
                AssetEvent::Removed { id } => {
                    changed_assets.remove(id);
                    removed.push(*id);
                }
                AssetEvent::LoadedWithDependencies { .. } => {
                    // TODO: handle this
                }
            }
        }

        let mut extracted_assets = Vec::new();
        for id in changed_assets.drain() {
            if let Some(asset) = assets.get(id) {
                extracted_assets.push((id, asset.clone()));
            }
        }

        commands.insert_resource(ExtractedMaterials {
            extracted: extracted_assets,
            removed,
        });
    }

    #[derive(Resource)]
    pub struct ChunkPipeline {
        mesh_pipeline: MeshPipeline,
        material_bind_group_layout: BindGroupLayout,
        world_bind_group_layout: BindGroupLayout,
        chunk_fragment_shader: Handle<Shader>,
    }
    impl FromWorld for ChunkPipeline {
        fn from_world(world: &mut bevy::prelude::World) -> Self {
            let world_data = world.resource::<WorldData>();
            let asset_server = world.resource::<AssetServer>();
            let render_device = world.resource::<RenderDevice>();
            let chunk_fragment_shader = asset_server.load("shaders/chunk.wgsl");
            Self {
                mesh_pipeline: world.resource::<MeshPipeline>().clone(),
                world_bind_group_layout: world_data.render_pass_bind_group_layout.clone(),
                material_bind_group_layout: ChunkMaterial::bind_group_layout(render_device),
                chunk_fragment_shader,
            }
        }
    }
    impl SpecializedMeshPipeline for ChunkPipeline {
        type Key = MeshPipelineKey;

        fn specialize(
            &self,
            key: Self::Key,
            layout: &MeshVertexBufferLayout,
        ) -> Result<RenderPipelineDescriptor, SpecializedMeshPipelineError> {
            let mut desc = self.mesh_pipeline.specialize(key, layout)?;
            // desc.primitive.cull_mode = Some(Face::Front);
            desc.primitive.cull_mode = Some(Face::Back);
            desc.layout
                .insert(1, self.material_bind_group_layout.clone());
            desc.layout.insert(3, self.world_bind_group_layout.clone());
            let frag = desc.fragment.as_mut().unwrap();
            frag.shader = self.chunk_fragment_shader.clone();
            // frag.shader_defs.push(ShaderDefVal::Bool("CHUNK_DEPTH_PREPASS".into(), false));
            Ok(desc)
        }
    }
    #[derive(Resource)]
    pub struct ChunkDepthPipeline {
        mesh_pipeline: MeshPipeline,
        material_bind_group_layout: BindGroupLayout,
        world_bind_group_layout: BindGroupLayout,
        chunk_fragment_shader: Handle<Shader>,
    }
    impl FromWorld for ChunkDepthPipeline {
        fn from_world(world: &mut bevy::prelude::World) -> Self {
            let world_data = world.resource::<WorldData>();
            let asset_server = world.resource::<AssetServer>();
            let render_device = world.resource::<RenderDevice>();
            let chunk_fragment_shader = asset_server.load("shaders/chunk.wgsl");
            Self {
                mesh_pipeline: world.resource::<MeshPipeline>().clone(),
                world_bind_group_layout: world_data.depth_pass_bind_group_layout.clone(),
                material_bind_group_layout: ChunkMaterial::bind_group_layout(render_device),
                chunk_fragment_shader,
            }
        }
    }
    impl SpecializedMeshPipeline for ChunkDepthPipeline {
        type Key = MeshPipelineKey;

        fn specialize(
            &self,
            key: Self::Key,
            layout: &MeshVertexBufferLayout,
        ) -> Result<RenderPipelineDescriptor, SpecializedMeshPipelineError> {
            let mut desc = self.mesh_pipeline.specialize(key, layout)?;
            desc.primitive.cull_mode = Some(Face::Back);
            desc.layout
                .insert(1, self.material_bind_group_layout.clone());
            // TODO: don't need this in depth prepass
            desc.layout.insert(3, self.world_bind_group_layout.clone());
            let frag = desc.fragment.as_mut().unwrap();
            frag.shader = self.chunk_fragment_shader.clone();
            frag.shader_defs.push(ShaderDefVal::Bool("CHUNK_DEPTH_PREPASS".into(), true));
            frag.targets[0] = Some(ColorTargetState {
                format: TextureFormat::R32Float,
                blend: None,
                write_mask: ColorWrites::RED,
            });
            Ok(desc)
        }
    }

    type ChunkDepthPrepass<M> = (
        SetItemPipeline,
        SetMeshViewBindGroup<0>,
        SetMaterialBindGroup<M, 1>,
        SetMeshBindGroup<2>,
        SetWorldDepthPassBindGroup<3>,
        DrawMesh,
    );
    type DrawMaterial<M> = (
        SetItemPipeline,
        SetMeshViewBindGroup<0>,
        SetMaterialBindGroup<M, 1>,
        SetMeshBindGroup<2>,
        SetWorldRenderPassBindGroup<3>,
        DrawMesh,
    );
    struct SetWorldRenderPassBindGroup<const I: usize>;
    impl<P: PhaseItem, const I: usize> RenderCommand<P> for SetWorldRenderPassBindGroup<I> {
        type Param = SRes<WorldData>;

        type ViewWorldQuery = ();

        type ItemWorldQuery = ();

        fn render<'w>(
            item: &P,
            view: ROQueryItem<'w, Self::ViewWorldQuery>,
            entity: ROQueryItem<'w, Self::ItemWorldQuery>,
            param: SystemParamItem<'w, '_, Self::Param>,
            pass: &mut TrackedRenderPass<'w>,
        ) -> RenderCommandResult {
            let world = param.into_inner();

            pass.set_bind_group(I, &world.render_pass_bind_group, &[]);

            RenderCommandResult::Success
        }
    }
    struct SetWorldDepthPassBindGroup<const I: usize>;
    impl<P: PhaseItem, const I: usize> RenderCommand<P> for SetWorldDepthPassBindGroup<I> {
        type Param = SRes<WorldData>;

        type ViewWorldQuery = ();

        type ItemWorldQuery = ();

        fn render<'w>(
            item: &P,
            view: ROQueryItem<'w, Self::ViewWorldQuery>,
            entity: ROQueryItem<'w, Self::ItemWorldQuery>,
            param: SystemParamItem<'w, '_, Self::Param>,
            pass: &mut TrackedRenderPass<'w>,
        ) -> RenderCommandResult {
            let world = param.into_inner();

            pass.set_bind_group(I, &world.depth_pass_bind_group, &[]);

            RenderCommandResult::Success
        }
    }
}

mod chunk {
    use bevy::pbr::DefaultOpaqueRendererMethod;
    use bevy::render::extract_component::ExtractComponent;
    use bevy::render::mesh::{Indices, VertexAttributeValues};
    use bevy::render::render_resource::{
        Extent3d, PrimitiveTopology, TextureDimension, TextureFormat,
    };
    use bevy::render::texture::TextureFormatPixelInfo;
    use bevy::tasks::{block_on, AsyncComputeTaskPool, Task};
    use bevy::utils::HashMap;
    use block_mesh::ndshape::ConstShape3u32;
    use block_mesh::{greedy_quads, GreedyQuadsBuffer, MergeVoxel, Voxel, VoxelVisibility};
    use noise::NoiseFn;

    use crate::player::PlayerEntity;

    use super::*;

    pub struct VoxelPlugin;
    impl Plugin for VoxelPlugin {
        fn build(&self, app: &mut App) {
            app.add_systems(
                Startup,
                (
                    setup_voxel_plugin,
                    // test_voxels,
                ),
            )
            // .add_systems(OnEnter(AppState::Playing), test_spawn)
            // .insert_resource(DefaultOpaqueRendererMethod::deferred())
            .add_systems(
                Update,
                (
                    || {},
                    // spawn_chunk_tasks,
                    // resolve_chunk_tasks,
                ),
            );
        }
    }

    fn test_voxels(
        mut commands: Commands,
        mut images: ResMut<Assets<Image>>,
        mut chunk_materials: ResMut<Assets<ChunkMaterial>>,
        mut meshes: ResMut<Assets<Mesh>>,
    ) {
        let scene_entity = commands
            .spawn((
                Name::new("test voxels"),
                GlobalTransform::default(),
                Transform::from_translation(Vec3::new(0.0, -500.0, 0.0)),
                VisibilityBundle::default(),
            ))
            .id();

        let size = 16.0;
        let side = 128usize;
        let lim = 512;
        let k = 0.09;
        let perlin = noise::Perlin::new(234342);

        let mut tc = TiledChunker::with_chunk_size(side);
        for z in 0..lim {
            for y in 0..lim {
                for x in 0..lim {
                    let pos = IVec3::new(x, y, z);

                    // let val = perlin.get([x as f64 * k, y as f64 * k, z as f64 * k]);
                    let val = rand::random::<f32>();
                    if val < 0.001 {
                        tc.set_voxel(pos, rand::random()).expect("should not fail");
                    }
                }
            }
            dbg!(z);
        }
        let materials = std::iter::once(Vec4::splat(0.0))
            .chain((1..256).map(|_| {
                Vec4::new(
                    rand::random::<f32>() * 0.5 + 0.2,
                    rand::random::<f32>() * 0.5 + 0.2,
                    rand::random::<f32>() * 0.5 + 0.2,
                    1.0,
                )
            }))
            .collect();

        let material_handle = images.add(Chunk::material_image(materials));
        let cube_handle = meshes.add(Mesh::from(shape::Cube { size: size * 2.0 }));
        for (chunk_index, chunk) in tc.chunks.into_iter() {
            let chunk_pos = Vec3::new(chunk_index.x as _, chunk_index.y as _, chunk_index.z as _);
            let chunk_pos = chunk_pos * size * 2.0;
            // + tc.chunk_side as f32 / 2.0;

            let mip1 = chunk.mip();
            let mip2 = mip1.mip();
            let voxels_handle = images.add(chunk.to_image());
            let mip1_handle = images.add(mip1.into_image());
            let mip2_handle = images.add(mip2.into_image());
            let chunk = Chunk {
                voxels: voxels_handle.clone(),
                materials: material_handle.clone(),
                side,
            };

            let chunk_material = chunk_materials.add(ChunkMaterial {
                side: side as _,
                voxels: voxels_handle.clone(),
                materials: material_handle.clone(),
                chunk_position: chunk_pos,
                chunk_size: size,
                voxels_mip1: mip1_handle,
                voxels_mip2: mip2_handle,
            });

            commands.entity(scene_entity).with_children(|builder| {
                builder.spawn((
                    Name::new("test voxel chunk"),
                    MaterialMeshBundle {
                        mesh: cube_handle.clone(),
                        material: chunk_material,
                        transform: Transform::from_translation(chunk_pos).with_scale(Vec3::NEG_ONE),
                        ..Default::default()
                    },
                    chunk,
                ));
            });
        }
    }

    #[derive(Resource)]
    pub struct VoxelWorld {
        noise: noise::Perlin,
    }

    pub struct ChunkFetchInfo<'a> {
        pos: Vec3,
        size: f32,
        chunk_entity: &'a mut Entity,
    }

    #[derive(Component, Clone, Debug)]
    pub enum SparseVoxelOctree {
        Root {
            position: Vec3,
            size: f32, // from centre to each side
            levels: usize,
            voxels: [Option<Box<Self>>; 8],
        },
        Node {
            voxels: [Option<Box<Self>>; 8],
        },
        Leaf {
            // 1 chunk = (8*16)^3 minecraft blocks = 2M
            // 1 minecraft block = (16 voxels)^3
            // 1 block = 1m
            // 1 chunk = 8m
            entity: Entity,
            // chunk: DefaultChunk,
        },
    }
    impl SparseVoxelOctree {
        fn root() -> Self {
            Self::Root {
                position: Vec3::ZERO,
                size: 128.0 / 2.0,
                voxels: Default::default(),
                levels: 4, // root level 1, chunk level 4
            }
        }

        fn get_mut_chunk(&mut self, mut pos: Vec3) -> Option<ChunkFetchInfo> {
            let Self::Root {
                position,
                size,
                voxels,
                levels,
            } = self
            else {
                unreachable!();
            };
            // dbg!(pos);
            // dbg!(&voxels);
            pos -= *position;
            let size = *size;
            pos /= size;
            if (pos.abs() - Vec3::ONE).max_element() < 0.0 {
                let mask = if pos.z > 0.0 { 0b100 } else { 0b000 }
                    | if pos.y > 0.0 { 0b010 } else { 0b000 }
                    | if pos.x > 0.0 { 0b001 } else { 0b000 };

                if voxels[mask].is_none() {
                    voxels[mask] = Some(Box::new(Self::Leaf {
                        // chunk: Default::default(),
                        entity: Entity::PLACEHOLDER,
                    }));
                    return voxels[mask].as_mut().map(|t| match &mut *(*t) {
                        Self::Leaf { entity } => ChunkFetchInfo {
                            pos: *position + (pos.signum() * size / 2.0),
                            size: size / 2.0,
                            // chunk,
                            chunk_entity: entity,
                        },
                        _ => unreachable!(),
                    });
                }
            }
            // dbg!();
            None
        }
    }

    pub struct TiledChunker {
        pub chunk_side: usize,
        pub chunks: HashMap<IVec3, ByteChunk>,
    }

    impl TiledChunker {
        pub fn with_chunk_size(size: usize) -> Self {
            Self {
                chunk_side: size,
                chunks: Default::default(),
            }
        }

        // https://www.desmos.com/calculator/dcrznftoqg
        pub fn get_voxel_index(&self, mut index: IVec3) -> UVec3 {
            let side = self.chunk_side as i32;
            // if index.x%side == 0 || index.z%side == 0 {
            //     let index = ((index % side) + side) % side;
            //     return  UVec3::new(side as u32 - 1, index.y as _, side as u32 - 1);
            // }
            let index = index.signum().min(IVec3::splat(0)) + index;
            let index = ((index % side) + side) % side;
            // let index = Vec3::new(index.x as _, index.y as _, index.z as _) % side as f32;
            // let index = index + side as f32;
            // let index = index % side as f32;
            UVec3::new(index.x as _, index.y as _, index.z as _)
        }
        pub fn get_chunk_index(&self, index: IVec3) -> IVec3 {
            let side = self.chunk_side as i32;
            let negative_offset = index.signum().min(IVec3::new(0, 0, 0));
            // floor((-1, -1)/2) gives (-1, -1)
            // but only in floating point math.
            // we need to subtract 1 in the directions with -ve values when working with integers
            negative_offset + index / side
            // let index = index.signum().min(IVec3::splat(0)) + index;
            // index / side
        }

        pub fn set_voxel(&mut self, index: IVec3, voxel: u8) -> Option<u8> {
            // let index = index.signum().min(IVec3::splat(0)) + index;
            let i1 = self.get_chunk_index(index);
            let index = self.get_voxel_index(index);
            match self.chunks.get_mut(&i1) {
                Some(c) => c.set(index, voxel),
                None => {
                    let mut c = ByteChunk::new_with_size(self.chunk_side);
                    let v = c.set(index, voxel);
                    self.chunks.insert(i1, c);
                    v
                }
            }
        }
    }

    #[derive(Clone, Debug, Default)]
    pub struct StaticChunkOctreeNode<T> {
        pub side: usize,
        pub subchunks: [Option<T>; 8],
    }

    pub type L1ChunkOctree = StaticChunkOctreeNode<ByteChunk>;
    pub type L2ChunkOctree = StaticChunkOctreeNode<StaticChunkOctreeNode<ByteChunk>>;

    impl L1ChunkOctree {
        pub fn set_voxel(&mut self, index: UVec3, voxel: u8) -> Option<u8> {
            let hside = self.side as u32 / 2;
            let i1 = UVec3::new(index.x % hside, index.y % hside, index.z % hside);
            match self.get_mut(index) {
                Some(c) => {
                    let c = c.get_or_insert_with(|| ByteChunk::new_with_size(hside as _));
                    c.set(i1, voxel)
                }
                None => None,
            }
        }
    }
    impl L2ChunkOctree {
        pub fn set_voxel(&mut self, index: UVec3, voxel: u8) -> Option<u8> {
            let hside = self.side as u32 / 2;
            let i1 = UVec3::new(index.x % hside, index.y % hside, index.z % hside);
            match self.get_mut(index) {
                Some(c) => {
                    let c = c.get_or_insert_with(|| L1ChunkOctree::new_with_size(hside as _));
                    c.set_voxel(i1, voxel)
                }
                None => None,
            }
        }
    }

    pub trait ChunkTrait {
        type Voxel;
        fn new_with_size(size: usize) -> Self;
        fn get_voxels(&self) -> &[Self::Voxel];
        fn get_voxels_mut(&mut self) -> &mut [Self::Voxel];
        fn get_index(&self, index: UVec3) -> Option<usize>;

        fn get_mut(&mut self, index: UVec3) -> Option<&mut Self::Voxel> {
            let index = self.get_index(index);
            index.map(|i| &mut self.get_voxels_mut()[i])
        }
        fn get(&self, index: UVec3) -> Option<&Self::Voxel> {
            let index = self.get_index(index);
            index.map(|i| &self.get_voxels()[i])
        }
        // returns old voxel at that position
        // returns None if index is outside the chunk
        fn set(&mut self, index: UVec3, voxel: Self::Voxel) -> Option<Self::Voxel>
        where
            Self::Voxel: Copy,
        {
            // if index.max_element() > self.side as _ {
            //     return None;
            // }
            // let side = self.side as u32;
            // let index = side.pow(2) * index.z + side * index.y + index.x;
            // let index = index as usize;
            // let old_voxel = self.voxels[index];
            // self.voxels[index] = voxel;
            // Some(old_voxel)
            match self.get_mut(index) {
                Some(old) => {
                    let old_voxel = *old;
                    *old = voxel;
                    Some(old_voxel)
                }
                None => None,
            }
        }
    }

    impl<T> ChunkTrait for StaticChunkOctreeNode<T> {
        type Voxel = Option<T>;

        fn new_with_size(side: usize) -> Self {
            Self {
                side,
                subchunks: Default::default(),
            }
        }

        fn get_voxels(&self) -> &[Self::Voxel] {
            &self.subchunks
        }

        fn get_voxels_mut(&mut self) -> &mut [Self::Voxel] {
            &mut self.subchunks
        }

        #[allow(clippy::identity_op)]
        fn get_index(&self, index: UVec3) -> Option<usize> {
            let side = self.side as _;
            if index.max_element() >= side {
                return None;
            }
            let hside = side / 2;
            let index = UVec3::new(
                ((index.z > hside) as u32) << 0b100,
                ((index.y > hside) as u32) << 0b010,
                ((index.x > hside) as u32) << 0b000,
            );
            let index = side.pow(2) * index.z + side * index.y + index.x;
            Some(index as _)
        }
    }

    impl ChunkTrait for ByteChunk {
        type Voxel = u8;

        fn new_with_size(side: usize) -> Self {
            Self {
                side,
                voxels: vec![0; side.pow(3)],
            }
        }

        fn get_voxels(&self) -> &[Self::Voxel] {
            &self.voxels
        }

        fn get_voxels_mut(&mut self) -> &mut [Self::Voxel] {
            &mut self.voxels
        }

        fn get_index(&self, index: UVec3) -> Option<usize> {
            let side = self.side as _;
            if index.max_element() >= side {
                return None;
            }
            let index = side.pow(2) * index.z + side * index.y + index.x;
            Some(index as _)
        }
    }

    #[derive(Clone, Debug)]
    pub struct U8VoxelChunk {
        pub voxels: Vec<U8Voxel>,
        pub side: usize,
    }

    impl U8VoxelChunk {
        pub fn byte_chunk(&self) -> ByteChunk {
            let side = self.side;
            let pside = side + 2;
            let mut new_voxels = vec![0; side.pow(3)];
            for z in 0..side {
                for y in 0..side {
                    for x in 0..side {
                        new_voxels[side.pow(2) * z + side * y + x] =
                            self.voxels[pside.pow(2) * (z + 1) + pside * (y + 1) + (x + 1)].0;
                    }
                }
            }

            ByteChunk {
                side,
                voxels: new_voxels,
            }
        }
    }

    #[derive(Clone, Debug)]
    pub struct ByteChunk {
        pub voxels: Vec<u8>,
        pub side: usize,
    }

    impl ByteChunk {
        #[allow(clippy::identity_op)]
        pub fn mip1_bytechunk(&self) -> Self {
            assert_eq!(self.side % 2, 0, "side should be divisible by 2");
            let new_side = self.side / 2;
            let mut bc = Self {
                voxels: vec![0; new_side.pow(3)],
                side: new_side,
            };
            let voxel = |z, y, x| self.voxels[z * self.side.pow(2) + y * self.side + x];

            for z in 0..new_side {
                for y in 0..new_side {
                    for x in 0..new_side {
                        let mut voxels = [(0u8, 0u8); 8];
                        let samples = [
                            voxel(z * 2 + 0, y * 2 + 0, x * 2 + 0),
                            voxel(z * 2 + 0, y * 2 + 0, x * 2 + 1),
                            voxel(z * 2 + 0, y * 2 + 1, x * 2 + 0),
                            voxel(z * 2 + 0, y * 2 + 1, x * 2 + 1),
                            voxel(z * 2 + 1, y * 2 + 0, x * 2 + 0),
                            voxel(z * 2 + 1, y * 2 + 0, x * 2 + 1),
                            voxel(z * 2 + 1, y * 2 + 1, x * 2 + 0),
                            voxel(z * 2 + 1, y * 2 + 1, x * 2 + 1),
                        ];
                        for sample in samples {
                            for i in 0..8 {
                                if voxels[i].0 == 0 {
                                    voxels[i] = (sample, 1);
                                    break;
                                } else if voxels[i].0 == sample {
                                    voxels[i].1 += 1;
                                    if voxels[i].1 > voxels[0].1 {
                                        voxels.swap(0, i);
                                    }
                                    break;
                                }
                            }
                        }
                        bc.voxels[bc.side.pow(2) * z + bc.side * y + x] = voxels[0].0;
                    }
                }
            }

            bc
        }

        // chunk side should be a multiple of 4
        // 1 byte stores 2x2x2 voxels each a single bit
        // 1 vec2<u32> stores 2x2x2 bytes, so 4x4x4 voxels
        // can use TextureFormat::Rg32Uint
        #[allow(clippy::erasing_op, clippy::identity_op)]
        pub fn mip(&self) -> MipChunk {
            let side = self.side;
            let new_side = side / 4;
            assert_eq!(side % 4, 0, "side should be a multiple of 4");

            let voxel = |z, y, x| (self.voxels[z * side.pow(2) + y * side + x] != 0) as u32;
            let byte = |z, y, x| {
                000 | voxel(z + 0, y + 0, x + 0) << 0b000
                    | voxel(z + 0, y + 0, x + 1) << 0b001
                    | voxel(z + 0, y + 1, x + 0) << 0b010
                    | voxel(z + 0, y + 1, x + 1) << 0b011
                    | voxel(z + 1, y + 0, x + 0) << 0b100
                    | voxel(z + 1, y + 0, x + 1) << 0b101
                    | voxel(z + 1, y + 1, x + 0) << 0b110
                    | voxel(z + 1, y + 1, x + 1) << 0b111
            };
            let chunk1 = |z, y, x| {
                000 | byte(z * 4 + 0, y * 4 + 0, x * 4 + 0) << (0b000 * 8)
                    | byte(z * 4 + 0, y * 4 + 0, x * 4 + 2) << (0b001 * 8)
                    | byte(z * 4 + 0, y * 4 + 2, x * 4 + 0) << (0b010 * 8)
                    | byte(z * 4 + 0, y * 4 + 2, x * 4 + 2) << (0b011 * 8)
            };
            let chunk2 = |z, y, x| {
                000 | byte(z * 4 + 2, y * 4 + 0, x * 4 + 0) << (0b000 * 8)
                    | byte(z * 4 + 2, y * 4 + 0, x * 4 + 2) << (0b001 * 8)
                    | byte(z * 4 + 2, y * 4 + 2, x * 4 + 0) << (0b010 * 8)
                    | byte(z * 4 + 2, y * 4 + 2, x * 4 + 2) << (0b011 * 8)
            };

            let mut mip = vec![UVec2::ZERO; new_side.pow(3)];
            for z in 0..new_side {
                for y in 0..new_side {
                    for x in 0..new_side {
                        mip[new_side.pow(2) * z + new_side * y + x] =
                            UVec2::new(chunk1(z, y, x), chunk2(z, y, x));
                    }
                }
            }
            MipChunk { voxels: mip, side }
        }

        pub fn to_image(self) -> Image {
            assert_eq!(self.side % 4, 0, "side should be a multiple of 4");
            Image::new(
                Extent3d {
                    width: self.side as u32 / 4,
                    height: self.side as _,
                    depth_or_array_layers: self.side as _,
                },
                TextureDimension::D3,
                self.voxels,
                TextureFormat::Rgba8Uint,
            )
        }
    }

    #[derive(Clone, Debug)]
    pub struct MipChunk {
        // voxels at the highest mip level
        pub side: usize,

        // length of this should always be (side/4)^3
        // each UVec2 stores 4x4x4 voxels
        pub voxels: Vec<UVec2>,
    }

    impl MipChunk {
        // from mip (0, 1, 2) to (3, 4, 5)
        // 0 -> 1 bit
        // 1 -> 1 byte, 2x2x2 voxels
        // 2 -> 2 x u32, 4x4x4 voxels
        #[allow(clippy::erasing_op, clippy::identity_op)]
        pub fn mip(&self) -> MipChunk {
            let mip_side = self.side / 4;
            let new_side = mip_side / 8;
            assert_eq!(self.side % (4 * 8), 0, "side should be a multiple of 32");

            let bit = |z, y, x| {
                let chunk: &UVec2 = &self.voxels[z * mip_side.pow(2) + y * mip_side + x];
                chunk.x != 0 || chunk.y != 0
            };
            let voxel = |z, y, x| {
                000 | (bit(z + 0, y + 0, x + 0)
                    || bit(z + 0, y + 0, x + 1)
                    || bit(z + 0, y + 1, x + 0)
                    || bit(z + 0, y + 1, x + 1)
                    || bit(z + 1, y + 0, x + 0)
                    || bit(z + 1, y + 0, x + 1)
                    || bit(z + 1, y + 1, x + 0)
                    || bit(z + 1, y + 1, x + 1)) as u32
            };
            let byte = |z, y, x| {
                000 | voxel(z + 0, y + 0, x + 0) << 0b000
                    | voxel(z + 0, y + 0, x + 2) << 0b001
                    | voxel(z + 0, y + 2, x + 0) << 0b010
                    | voxel(z + 0, y + 2, x + 2) << 0b011
                    | voxel(z + 2, y + 0, x + 0) << 0b100
                    | voxel(z + 2, y + 0, x + 2) << 0b101
                    | voxel(z + 2, y + 2, x + 0) << 0b110
                    | voxel(z + 2, y + 2, x + 2) << 0b111
            };
            let chunk1 = |z, y, x| {
                000 | byte(z * 8 + 0, y * 8 + 0, x * 8 + 0) << (0b000 * 8)
                    | byte(z * 8 + 0, y * 8 + 0, x * 8 + 4) << (0b001 * 8)
                    | byte(z * 8 + 0, y * 8 + 4, x * 8 + 0) << (0b010 * 8)
                    | byte(z * 8 + 0, y * 8 + 4, x * 8 + 4) << (0b011 * 8)
            };
            let chunk2 = |z, y, x| {
                000 | byte(z * 8 + 4, y * 8 + 0, x * 8 + 0) << (0b000 * 8)
                    | byte(z * 8 + 4, y * 8 + 0, x * 8 + 4) << (0b001 * 8)
                    | byte(z * 8 + 4, y * 8 + 4, x * 8 + 0) << (0b010 * 8)
                    | byte(z * 8 + 4, y * 8 + 4, x * 8 + 4) << (0b011 * 8)
            };

            let mut mip = vec![UVec2::ZERO; new_side.pow(3)];
            for z in 0..new_side {
                for y in 0..new_side {
                    for x in 0..new_side {
                        mip[new_side.pow(2) * z + new_side * y + x] =
                            UVec2::new(chunk1(z, y, x), chunk2(z, y, x));
                    }
                }
            }
            MipChunk {
                voxels: mip,
                side: mip_side / 2,
            }
        }

        pub fn into_image(self) -> Image {
            let mip_side = self.side / 4;
            Image::new(
                Extent3d {
                    width: mip_side as _,
                    height: mip_side as _,
                    depth_or_array_layers: mip_side as _,
                },
                TextureDimension::D3,
                self.voxels
                    .into_iter()
                    .flat_map(|v| [v.x.to_le_bytes(), v.y.to_le_bytes()])
                    .flatten()
                    .collect(),
                TextureFormat::Rg32Uint,
            )
        }
    }

    pub const DEFAULT_CHUNK_SIDE: u32 = 8 * 16;
    pub const PADDED_DEFAULT_CHUNK_SIDE: u32 = 8 * 16 + 2;

    #[derive(Debug, Clone, Copy, Eq, PartialEq)]
    pub struct U8Voxel(u8);
    impl Voxel for U8Voxel {
        fn get_visibility(&self) -> VoxelVisibility {
            if self.0 == 0 {
                VoxelVisibility::Empty
            } else {
                VoxelVisibility::Opaque
            }
        }
    }
    impl MergeVoxel for U8Voxel {
        type MergeValue = Self;

        fn merge_value(&self) -> Self::MergeValue {
            *self
        }
    }

    #[derive(Component, Clone, Debug)]
    pub struct Chunk {
        // TODO: turn these to have a runtime parameter of size: UVec3
        pub side: usize,
        pub voxels: Handle<Image>,
        pub materials: Handle<Image>,
    }

    /// N should be a multiple of 4
    impl Chunk {
        pub fn empty(assets: &mut Assets<Image>) -> Self {
            Self {
                voxels: assets.add(
                    ByteChunk {
                        voxels: vec![0; DEFAULT_CHUNK_SIDE.pow(3) as _],
                        side: DEFAULT_CHUNK_SIDE as _,
                    }
                    .to_image(),
                ),
                materials: assets.add(Self::material_image(vec![Default::default(); 256])),
                side: DEFAULT_CHUNK_SIDE as _,
            }
        }

        pub fn material_image(materials: Vec<Vec4>) -> Image {
            Image::new(
                Extent3d {
                    width: 256,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                TextureDimension::D1,
                materials
                    .into_iter()
                    .flat_map(|v| [v.x, v.y, v.z, v.w])
                    .map(|v| (v * 255.0) as u8)
                    .collect(),
                TextureFormat::Rgba8Unorm, // Rgba8UnormSrgb is gamma-corrected
            )
        }
    }

    // - [AsBindGroup in bevy::render::render_resource - Rust](https://docs.rs/bevy/latest/bevy/render/render_resource/trait.AsBindGroup.html)
    #[derive(Asset, TypePath, AsBindGroup, Debug, Clone, Component, ExtractComponent)]
    pub struct ChunkMaterial {
        #[uniform(0)]
        pub side: u32,
        #[texture(2, dimension = "1d", sample_type = "float", filterable = false)]
        pub materials: Handle<Image>,
        #[uniform(5)]
        pub chunk_position: Vec3,
        #[uniform(6)]
        pub chunk_size: f32,
        #[texture(1, sample_type = "u_int", dimension = "3d")]
        pub voxels: Handle<Image>,
        #[texture(7, sample_type = "u_int", dimension = "3d")]
        pub voxels_mip1: Handle<Image>,
        #[texture(8, sample_type = "u_int", dimension = "3d")]
        pub voxels_mip2: Handle<Image>,
    }

    impl Material for ChunkMaterial {
        fn fragment_shader() -> ShaderRef {
            "shaders/chunk.wgsl".into()
        }

        fn alpha_mode(&self) -> AlphaMode {
            AlphaMode::Opaque
        }
    }

    #[derive(Component)]
    pub struct ChunkSpawnTask {
        task: Task<((U8VoxelChunk, Vec3, f32), Mesh)>,
    }

    fn setup_voxel_plugin(mut commands: Commands, mut game_state: ResMut<NextState<AppState>>) {
        let perlin = noise::Perlin::new(3243);
        let world = VoxelWorld { noise: perlin };
        commands.insert_resource(world);
        game_state.set(AppState::Playing);

        // commands.spawn(DirectionalLightBundle {
        //     directional_light: DirectionalLight {
        //         illuminance: 3000.0,
        //         shadows_enabled: true,
        //         ..Default::default()
        //     },
        //     transform: Transform::from_xyz(1.0, 1.0, 1.0)
        //         .with_rotation(Quat::from_axis_angle(Vec3::X, 3.5)),
        //     ..Default::default()
        // });
        // commands.spawn((
        //     SparseVoxelOctree::root(),
        //     TransformBundle::default(),
        //     VisibilityBundle::default(),
        // ));
    }

    /*
    fn spawn_chunk_tasks(
        mut commands: Commands,
        world: Res<VoxelWorld>,
        mut svo: Query<(Entity, &mut SparseVoxelOctree)>,
        players: Query<(&PlayerEntity, &Transform)>,
    ) {
        let threadpool = AsyncComputeTaskPool::get();

        let (_, player) = players.single();

        for (svo_entity, mut svo) in svo.iter_mut() {
            let Some(chunk) = svo.get_mut_chunk(player.translation) else {
                return;
            };
            dbg!("spawining shunk", chunk.pos, chunk.size);

            let size = chunk.size;
            let pos = chunk.pos;
            let perlin = world.noise;
            let task = threadpool.spawn(async move {
                let side = DEFAULT_CHUNK_SIDE as usize;
                let pside = PADDED_DEFAULT_CHUNK_SIDE as usize;
                let mut voxels = U8VoxelChunk(vec![U8Voxel(0); pside * pside * pside]);
                let scale = 0.09;
                for z in 0..side {
                    for x in 0..side {
                        let xf =
                            ((x as f32 - side as f32 / 2.0) / side as f32) * size * 2.0 + pos.x;
                        let zf =
                            ((z as f32 - side as f32 / 2.0) / side as f32) * size * 2.0 + pos.z;
                        let v = perlin.get([xf as f64 * scale, zf as f64 * scale]);

                        for y in 0..side {
                            let yf =
                                ((y as f32 - side as f32 / 2.0) / side as f32) * size * 2.0 + pos.y;
                            if (yf as f64 + 20.0) < v * 10.0 {
                                voxels.0[(z + 1) * pside * pside + (y + 1) * pside + (x + 1)] =
                                    U8Voxel(1);
                            }
                        }
                    }
                }
                let mut buffer = GreedyQuadsBuffer::new(voxels.0.len());
                type ChunkShape = ConstShape3u32<
                    PADDED_DEFAULT_CHUNK_SIDE,
                    PADDED_DEFAULT_CHUNK_SIDE,
                    PADDED_DEFAULT_CHUNK_SIDE,
                >;
                let faces = &block_mesh::RIGHT_HANDED_Y_UP_CONFIG.faces;
                greedy_quads(
                    &voxels.0,
                    &ChunkShape {},
                    [0; 3],
                    [DEFAULT_CHUNK_SIDE + 1; 3],
                    faces,
                    &mut buffer,
                );

                let num_indices = buffer.quads.num_quads() * 6;
                let num_vertices = buffer.quads.num_quads() * 4;
                let mut indices = Vec::with_capacity(num_indices);
                let mut positions = Vec::with_capacity(num_vertices);
                let mut normals = Vec::with_capacity(num_vertices);
                for (group, face) in buffer.quads.groups.into_iter().zip(faces.iter()) {
                    for quad in group.into_iter() {
                        indices.extend_from_slice(&face.quad_mesh_indices(positions.len() as u32));
                        positions.extend_from_slice(
                            &face
                                .quad_mesh_positions(&quad, 2.0 * size / DEFAULT_CHUNK_SIDE as f32),
                        );
                        normals.extend_from_slice(&face.quad_mesh_normals());
                    }
                }

                let mut render_mesh = Mesh::new(PrimitiveTopology::TriangleList);
                render_mesh.insert_attribute(
                    Mesh::ATTRIBUTE_POSITION,
                    VertexAttributeValues::Float32x3(positions),
                );
                render_mesh.insert_attribute(
                    Mesh::ATTRIBUTE_NORMAL,
                    VertexAttributeValues::Float32x3(normals),
                );
                render_mesh.insert_attribute(
                    Mesh::ATTRIBUTE_UV_0,
                    VertexAttributeValues::Float32x2(vec![[0.0; 2]; num_vertices]),
                );
                render_mesh.set_indices(Some(Indices::U32(indices.clone())));

                ((voxels, pos, size), render_mesh)
            });

            *chunk.chunk_entity = commands.spawn_empty().id();
            commands
                .entity(*chunk.chunk_entity)
                .insert(ChunkSpawnTask { task });

            commands.entity(svo_entity).add_child(*chunk.chunk_entity);
        }
    }

    fn resolve_chunk_tasks<const N: usize>(
        mut commands: Commands,
        mut meshes: ResMut<Assets<Mesh>>,
        mut materials: ResMut<Assets<StandardMaterial>>,
        mut chunk_materials: ResMut<Assets<ChunkMaterial>>,
        mut images: ResMut<Assets<Image>>,
        mut tasks: Query<(Entity, &mut ChunkSpawnTask<N>)>,
    ) {
        for (task_entity, mut task) in tasks.iter_mut() {
            // tasks can be cancelled. so it returns an Option
            if let Some(((voxels, pos, size), mesh)) =
                block_on(futures_lite::future::poll_once(&mut task.task))
            {
                let empty_mesh = mesh.count_vertices() == 0;

                let mesh_handle = meshes.add(mesh);
                let mut material_buffer = vec![Default::default(); 256];
                material_buffer[1] = Vec4::new(0.8, 0.8, 0.8, 1.0);
                // let chunk = DefaultChunk::from_u8voxels(&mut images, voxels, material_buffer);
                let chunk = DefaultChunk {
                    voxels: images.add(voxels.byte_chunk().to_image()),
                    materials: images.add(DefaultChunk::material_image(material_buffer)),
                };

                let r = (DEFAULT_CHUNK_SIDE as f32 + 2.0) / (DEFAULT_CHUNK_SIDE as f32);
                let bb_mesh_relative_pos = Vec3::ZERO;
                let voxel_mesh_relative_pos = Vec3::splat(-size * r);
                commands
                    .entity(task_entity)
                    .insert((
                        TransformBundle::from_transform(Transform::from_translation(pos)),
                        VisibilityBundle::default(),
                    ))
                    .remove::<ChunkSpawnTask<N>>();

                if empty_mesh {
                    continue;
                }

                commands
                    .entity(task_entity)
                    .with_children(|parent| {
                        parent.spawn((
                            MaterialMeshBundle {
                                mesh: meshes.add(Mesh::from(shape::Cube { size: size * 2.0 })),
                                material: chunk_materials.add(ChunkMaterial {
                                    side: DEFAULT_CHUNK_SIDE,
                                    voxels: chunk.voxels.clone(),
                                    materials: chunk.materials.clone(),
                                    player_position: Vec3::ZERO,
                                    resolution: Vec2::ZERO,
                                    chunk_position: Vec3::ZERO,
                                    chunk_size: size,
                                }),
                                visibility: Visibility::Visible,
                                transform: Transform::from_translation(bb_mesh_relative_pos),
                                ..Default::default()
                            },
                            Name::new("Cube Mesh"),
                        ));
                        parent.spawn((
                            MaterialMeshBundle {
                                mesh: mesh_handle.clone(),
                                // material: materials.add(StandardMaterial {
                                //     base_color: Color::rgb(0.8, 0.8, 0.8),
                                //     alpha_mode: AlphaMode::Opaque,
                                //     ..Default::default()
                                // }),
                                material: chunk_materials.add(ChunkMaterial {
                                    side: DEFAULT_CHUNK_SIDE,
                                    voxels: chunk.voxels.clone(),
                                    materials: chunk.materials.clone(),
                                    player_position: Vec3::ZERO,
                                    resolution: Vec2::ZERO,
                                    chunk_position: Vec3::ZERO,
                                    chunk_size: size,
                                }),
                                transform: Transform::from_translation(voxel_mesh_relative_pos),
                                visibility: Visibility::Hidden,
                                ..Default::default()
                            },
                            Name::new("Voxel Mesh"),
                        ));
                    })
                    .insert((Name::new("Chunk"), chunk));
            }
        }
    }
    */
}

mod vox {
    use bevy::{
        asset::{AssetLoader, AsyncReadExt},
        prelude::*,
        utils::HashMap,
    };
    use dot_vox::{DotVoxData, Model, SceneNode};

    use crate::chunk::{
        ByteChunk, Chunk, ChunkMaterial, ChunkTrait, L1ChunkOctree, L2ChunkOctree, TiledChunker,
        DEFAULT_CHUNK_SIDE,
    };

    #[derive(Component, Clone, Copy)]
    pub struct VoxChunk;

    #[derive(Component, Clone, Copy)]
    pub struct VoxScene;

    #[derive(Asset, TypePath)]
    pub struct Vox(DotVoxData);

    #[derive(Default)]
    pub struct VoxLoader;
    impl Plugin for VoxLoader {
        fn build(&self, app: &mut bevy::prelude::App) {
            app.init_asset_loader::<VoxLoader>()
                .init_asset::<Vox>()
                .add_systems(Startup, test_load)
                .add_systems(Update, on_vox_asset_event);
        }
    }

    fn test_load(asset_server: Res<AssetServer>, mut commands: Commands) {
        // TODO: decide what to do with this Vox oject. it is already loaded by VoxLoader
        commands.spawn((
            // asset_server.load::<Vox>("./castle_pro.vox"),
            asset_server.load::<Vox>("./castle.vox"),
            GlobalTransform::default(),
        ));
    }

    impl AssetLoader for VoxLoader {
        type Asset = Vox;
        type Settings = ();
        type Error = anyhow::Error;

        fn load<'a>(
            &'a self,
            reader: &'a mut bevy::asset::io::Reader,
            _settings: &'a Self::Settings,
            // NOTE: could use this to load DotVoxData directly into Handle<Image> and stuff,
            // but it will also need to spawn chunks, so not doing that here.
            _load_context: &'a mut bevy::asset::LoadContext,
        ) -> bevy::utils::BoxedFuture<'a, Result<Self::Asset, Self::Error>> {
            Box::pin(async {
                let mut data = vec![];
                reader.read_to_end(&mut data).await?;
                let data = dot_vox::load_bytes(&data).map_err(anyhow::Error::msg)?;
                Ok(Vox(data))
            })
        }

        fn extensions(&self) -> &[&str] {
            &["vox"]
        }
    }

    fn on_vox_asset_event(
        mut events: EventReader<AssetEvent<Vox>>,
        vox_assets: Res<Assets<Vox>>,
        mut commands: Commands,
        mut images: ResMut<Assets<Image>>,
        mut chunk_materials: ResMut<Assets<ChunkMaterial>>,
        mut meshes: ResMut<Assets<Mesh>>,
        mut std_materials: ResMut<Assets<StandardMaterial>>,
    ) {
        for event in events.read() {
            let AssetEvent::LoadedWithDependencies { id } = event else {
                continue;
            };
            let data = vox_assets.get(*id).expect("event said it is loaded");

            load_vox(
                &mut commands,
                &mut images,
                &mut chunk_materials,
                &mut meshes,
                &mut std_materials,
                &data.0,
            );
        }
    }

    struct VoxModel {
        model: Model,
        translation: IVec3,
        rotation: Quat,
        scale: Vec3,
    }
    struct VoxParser<'a> {
        data: &'a DotVoxData,
        models: Vec<VoxModel>,
    }
    impl VoxParser<'_> {
        fn parse(&mut self, scene_index: usize, translation: IVec3, rotation: Quat, scale: Vec3) {
            let scene = &self.data.scenes[scene_index];
            match scene {
                SceneNode::Transform { frames, child, .. } => {
                    assert_eq!(frames.len(), 1, "unimplemented");
                    let frame = &frames[0];
                    let translation = translation
                        + frame
                            .position()
                            .map(|p| IVec3::new(p.x as _, p.z as _, p.y as _))
                            .unwrap_or_default();
                    let (rotation, scale) = frame
                        .orientation()
                        .map(|o| o.to_quat_scale())
                        .map(|(q, f)| {
                            (
                                Quat::from_xyzw(q[0], q[2], -q[1], q[3]),
                                Vec3::new(f[0], f[2], f[1]),
                            )
                        })
                        .unwrap_or_default();

                    self.parse(*child as _, translation, rotation, scale);
                }
                SceneNode::Group { children, .. } => {
                    children.iter().for_each(|child| {
                        self.parse(*child as _, translation, rotation, scale);
                    });
                }
                SceneNode::Shape {
                    models: shape_models,
                    ..
                } => {
                    assert_eq!(shape_models.len(), 1, "unimplemented");
                    let model = &self.data.models[shape_models[0].model_id as usize];

                    // let mut offset = Vec3::new(
                    //     if model.size.x % 2 == 0 { 0.0 } else { 0.5 },
                    //     if model.size.z % 2 == 0 { 0.0 } else { 0.5 },
                    //     if model.size.y % 2 == 0 { 0.0 } else { -0.5 },
                    // );
                    // offset = rotation.mul_vec3(offset);
                    // let center = rotation
                    //     * (Vec3::new(model.size.x as _, model.size.y as _, model.size.z as _)
                    //         / 2.0);
                    // let translation = translation - center * scale + offset;

                    let vmodel = VoxModel {
                        model: Model {
                            voxels: model
                                .voxels
                                .iter()
                                .map(|v| dot_vox::Voxel {
                                    x: v.x,
                                    y: v.z,
                                    z: v.y,
                                    i: v.i + 1,
                                })
                                .collect(),
                            size: dot_vox::Size {
                                x: model.size.x,
                                y: model.size.z,
                                z: model.size.y,
                            },
                        },
                        translation,
                        rotation,
                        scale,
                    };
                    self.models.push(vmodel);
                }
            }
        }

        fn get_materials(&self) -> Vec<Vec4> {
            let materials = [Vec4::splat(0.0)]
                .into_iter()
                .chain(
                    self.data
                        .palette
                        .iter()
                        .map(|c| Vec4::new(c.r as f32, c.g as f32, c.b as f32, c.a as f32)),
                )
                .map(|c| c / 256.0)
                .take(256) // idk why this buffer has 1 extra material :/
                .collect::<Vec<_>>();
            materials
        }

        fn tiled_chunker(&self, chunk_side: usize) -> TiledChunker {
            let mut tc = TiledChunker::with_chunk_size(chunk_side);

            for model in self.models.iter() {
                let chunk_pos = model.translation;
                let pivot = Vec3::new(
                    (model.model.size.x / 2) as _,
                    (model.model.size.y / 2) as _,
                    (model.model.size.z / 2) as _,
                );
                // TODO: inserting each voxel using tc.set_voxel is gonna be slow as each voxel will require a hashmap.get
                for voxel in &model.model.voxels {
                    let mut voxel_pos = IVec3::new(voxel.x as _, voxel.y as _, voxel.z as _);

                    // if model.scale.x.is_sign_negative() {
                    //     voxel_pos.x = model.model.size.x as i32 - voxel.x as i32 - 1;
                    // }
                    // if model.scale.y.is_sign_negative() {
                    //     voxel_pos.x = model.model.size.y as i32 - voxel.y as i32 - 1;
                    // }
                    // if model.scale.z.is_sign_negative() {
                    //     voxel_pos.x = model.model.size.z as i32 - voxel.z as i32 - 1;
                    // }

                    // https://github.com/jpaver/opengametools/blob/a45c00f7c4ff2e7f750099cba7fcc299e0d08096/src/ogt_vox.h#L120
                    let voxel_pos =
                        Vec3::new(voxel_pos.x as _, voxel_pos.y as _, voxel_pos.z as _) - pivot;
                    let voxel_pos = model.rotation.mul_vec3(voxel_pos);
                    let voxel_pos = IVec3::new(
                        voxel_pos.x.round() as _,
                        voxel_pos.y.round() as _,
                        voxel_pos.z.round() as _,
                    );

                    let pivot = IVec3::new(
                        (model.model.size.x / 2) as _,
                        (model.model.size.y / 2) as _,
                        (model.model.size.z / 2) as _,
                    );
                    tc.set_voxel(chunk_pos + voxel_pos, voxel.i)
                        .expect("invalid index");
                }
            }

            // let keys = tc.chunks.iter().map(|(k, v)| k).collect::<Vec<_>>();
            // dbg!(keys);

            tc
        }
    }

    fn load_vox(
        commands: &mut Commands,
        images: &mut Assets<Image>,
        chunk_materials: &mut Assets<ChunkMaterial>,
        meshes: &mut Assets<Mesh>,
        std_materials: &mut Assets<StandardMaterial>,
        data: &DotVoxData,
    ) {
        let vox_scene_entity = commands
            .spawn((
                Name::new("VoxScene"),
                VoxScene,
                GlobalTransform::default(),
                Transform::from_translation(Vec3::new(0.0, 0.0, 0.0)),
                VisibilityBundle::default(),
            ))
            .id();

        let mut parser = VoxParser {
            data,
            models: vec![],
        };
        parser.parse(0, IVec3::default(), Quat::default(), Vec3::ZERO);
        // let models = parser
        //     .models
        //     .iter()
        //     .map(|m| {
        //         (
        //             m.translation,
        //             m.translation
        //                 + IVec3::new(
        //                     m.model.size.x as _,
        //                     m.model.size.z as _,
        //                     m.model.size.y as _,
        //                 ),
        //         )
        //     })
        //     .reduce(|a, b| (a.0.min(b.0), a.1.max(b.1)));
        // dbg!(models);

        let side = DEFAULT_CHUNK_SIDE as usize;
        // let side = 256;
        let size = 16.0;
        let tc = parser.tiled_chunker(side);
        let material_handle = images.add(Chunk::material_image(parser.get_materials()));
        let cube_handle = meshes.add(Mesh::from(shape::Cube { size: size * 2.0 }));
        for (chunk_index, chunk) in tc.chunks.into_iter() {
            let chunk_pos = Vec3::new(chunk_index.x as _, chunk_index.y as _, chunk_index.z as _);
            let chunk_pos = chunk_pos * size * 2.0;
            // + tc.chunk_side as f32 / 2.0;

            // let chunk = chunk.mip1_bytechunk();
            // let chunk = chunk.mip1_bytechunk();
            // let chunk = chunk.mip1_bytechunk();
            let side = chunk.side;

            let mip1 = chunk.mip();
            let mip2 = mip1.mip();
            let voxels_handle = images.add(chunk.to_image());
            let mip1_handle = images.add(mip1.into_image());
            let mip2_handle = images.add(mip2.into_image());
            let chunk = Chunk {
                voxels: voxels_handle.clone(),
                materials: material_handle.clone(),
                side,
            };

            let chunk_material = chunk_materials.add(ChunkMaterial {
                side: side as _,
                voxels: voxels_handle.clone(),
                materials: material_handle.clone(),
                chunk_position: chunk_pos,
                chunk_size: size,
                voxels_mip1: mip1_handle,
                voxels_mip2: mip2_handle,
            });

            commands.entity(vox_scene_entity).with_children(|builder| {
                builder.spawn((
                    Name::new("VoxChunk"),
                    VoxChunk,
                    MaterialMeshBundle {
                        mesh: cube_handle.clone(),
                        // material: std_materials.add(StandardMaterial {
                        //     base_color: Color::rgb(0.8, 0.8, 0.8),
                        //     alpha_mode: AlphaMode::Opaque,
                        //     ..Default::default()
                        // }),
                        material: chunk_material,
                        transform: Transform::from_translation(chunk_pos).with_scale(Vec3::NEG_ONE),
                        ..Default::default()
                    },
                    chunk,
                ));
            });
        }
        // todo!();

        // let side = DEFAULT_CHUNK_SIDE as usize;
        // let size = 16.0;

        // let materials = [Vec4::splat(0.0)]
        //     .into_iter()
        //     .chain(
        //         data.palette
        //             .iter()
        //             .map(|c| Vec4::new(c.r as f32, c.g as f32, c.b as f32, c.a as f32)),
        //     )
        //     .map(|c| c / 256.0)
        //     .take(256) // idk why this buffer has 1 extra material :/
        //     .collect::<Vec<_>>();
        // let material_handle = images.add(Chunk::material_image(materials));
        // let cube_handle = meshes.add(Mesh::from(shape::Cube { size: size * 2.0 }));

        // let mut chunks = HashMap::new();

        // // TODO: vox chunks can be of side 256, but this code crashes on chunk size > DEFAULT_CHUNK_SIDE
        // for (i, model) in data.models.iter().enumerate() {
        //     let mut voxels = ByteChunk {
        //         voxels: vec![0u8; side.pow(3)],
        //         side,
        //     };
        //     for voxel in &model.voxels {
        //         let _ = voxels.set(
        //             UVec3::new(voxel.x as _, voxel.z as _, voxel.y as _),
        //             voxel.i + 1,
        //         );
        //     }
        //     let mip1 = voxels.mip();
        //     let mip2 = mip1.mip();
        //     let voxels_handle = images.add(voxels.to_image());
        //     let mip1_handle = images.add(mip1.into_image());
        //     let mip2_handle = images.add(mip2.into_image());
        //     let chunk = Chunk {
        //         voxels: voxels_handle.clone(),
        //         materials: material_handle.clone(),
        //         side,
        //     };

        //     let chunk_material = chunk_materials.add(ChunkMaterial {
        //         side: side as _,
        //         voxels: voxels_handle.clone(),
        //         materials: material_handle.clone(),
        //         player_position: Vec3::ZERO,
        //         resolution: Vec2::ZERO,
        //         chunk_position: Vec3::ZERO,
        //         chunk_size: size,
        //         voxels_mip1: mip1_handle,
        //         voxels_mip2: mip2_handle,
        //     });

        //     commands.entity(vox_scene_entity).with_children(|castle| {
        //         let chunk_entity = castle
        //             .spawn((
        //                 Name::new("VoxChunk"),
        //                 VoxChunk,
        //                 MaterialMeshBundle {
        //                     mesh: cube_handle.clone(),
        //                     // material: std_materials.add(StandardMaterial {
        //                     //     base_color: Color::rgb(0.8, 0.8, 0.8),
        //                     //     alpha_mode: AlphaMode::Opaque,
        //                     //     ..Default::default()
        //                     // }),
        //                     material: chunk_material,
        //                     visibility: Visibility::Visible,
        //                     ..Default::default()
        //                 },
        //                 chunk,
        //             ))
        //             .id();

        //         chunks.insert(i, chunk_entity);
        //     });
        // }

        // #[allow(clippy::too_many_arguments)]
        // fn set_transforms(
        //     scene_index: usize,
        //     chunks: &HashMap<usize, Entity>,
        //     commands: &mut Commands,
        //     scenes: &[SceneNode],
        //     models: &[Model],
        //     translation: Vec3,
        //     rotation: Quat,
        //     scale: Vec3,
        //     voxel_size: f32,
        // ) {
        //     let scene = &scenes[scene_index];
        //     match scene {
        //         SceneNode::Transform { frames, child, .. } => {
        //             assert_eq!(frames.len(), 1, "unimplemented");
        //             let frame = &frames[0];
        //             let translation = translation
        //                 + frame
        //                     .position()
        //                     .map(|p| Vec3::new(p.x as _, p.z as _, p.y as _))
        //                     .unwrap_or_default();
        //             let (rotation, scale) = frame
        //                 .orientation()
        //                 .map(|o| o.to_quat_scale())
        //                 .map(|(q, f)| {
        //                     (Quat::from_xyzw(q[0], q[2], q[1], q[3]), Vec3::from_array(f))
        //                 })
        //                 .unwrap_or_default();

        //             set_transforms(
        //                 *child as _,
        //                 chunks,
        //                 commands,
        //                 scenes,
        //                 models,
        //                 translation,
        //                 rotation,
        //                 scale,
        //                 voxel_size,
        //             );
        //         }
        //         SceneNode::Group { children, .. } => {
        //             children.iter().for_each(|child| {
        //                 set_transforms(
        //                     *child as _,
        //                     chunks,
        //                     commands,
        //                     scenes,
        //                     models,
        //                     translation,
        //                     rotation,
        //                     scale,
        //                     voxel_size,
        //                 );
        //             });
        //         }
        //         SceneNode::Shape {
        //             models: shape_models,
        //             ..
        //         } => {
        //             assert_eq!(shape_models.len(), 1, "unimplemented");
        //             let Some(&chunk) = chunks.get(&(shape_models[0].model_id as usize)) else {
        //                 return;
        //             };
        //             let model = &models[shape_models[0].model_id as usize];

        //             let mut offset = Vec3::new(
        //                 if model.size.x % 2 == 0 { 0.0 } else { 0.5 },
        //                 if model.size.z % 2 == 0 { 0.0 } else { 0.5 },
        //                 if model.size.y % 2 == 0 { 0.0 } else { -0.5 },
        //             );
        //             offset = rotation.mul_vec3(offset);
        //             let center = rotation
        //                 * (Vec3::new(model.size.x as _, model.size.y as _, model.size.z as _)
        //                     / 2.0);
        //             let translation = translation - center * scale + offset;

        //             commands.entity(chunk).insert(
        //                 Transform::from_translation(translation * voxel_size)
        //                     .with_scale(Vec3::NEG_ONE)
        //                     .with_rotation(rotation), //.with_scale(scale),
        //             );
        //         }
        //     }
        // }

        // set_transforms(
        //     0,
        //     &chunks,
        //     commands,
        //     &data.scenes,
        //     &data.models,
        //     Vec3::default(),
        //     Quat::default(),
        //     Vec3::ZERO,
        //     size / 256.0, // 256 is the maximum chunk size in the vox format
        // );
    }
}

mod player {
    use super::*;

    #[derive(Component)]
    pub struct PlayerEntity;

    pub struct PlayerPlugin;
    impl Plugin for PlayerPlugin {
        fn build(&self, app: &mut App) {
            app.add_systems(Startup, setup)
                .add_systems(Update, update.run_if(in_state(ControlsState::Player)));
        }
    }

    fn setup(mut commands: Commands, mut _meshes: ResMut<Assets<Mesh>>) {
        let transform = Transform::from_xyz(5.0, 5.0, -5.0).looking_at(Vec3::ZERO, Vec3::Y);
        commands.spawn((
            Camera3dBundle {
                transform,
                ..default()
            },
            PlayerEntity,
        ));
    }

    fn update(
        mut player: Query<(&mut Transform, &mut PlayerEntity)>,
        keys: Res<Input<KeyCode>>,
        mouse_keys: Res<Input<MouseButton>>,
        mut mouse_events: EventReader<MouseMotion>,
        // time: Res<Time>,
        mut window: Query<&mut Window, With<PrimaryWindow>>,
    ) {
        let mut wind = window.single_mut();
        if keys.just_pressed(KeyCode::Escape) {
            wind.cursor.grab_mode = match wind.cursor.grab_mode {
                CursorGrabMode::Locked => {
                    wind.cursor.visible = true;
                    CursorGrabMode::None
                }
                _ => {
                    wind.cursor.visible = false;
                    CursorGrabMode::Locked
                }
            };
            let pos = Vec2::new(
                wind.resolution.width() / 2.0,
                wind.resolution.height() / 2.0,
            );
            wind.set_cursor_position(Some(pos));
            mouse_events.clear();
        }

        let (mut transform, _) = player.single_mut();
        if wind.cursor.grab_mode != CursorGrabMode::Locked {
            let pos = Vec2::new(
                wind.resolution.width() / 2.0,
                wind.resolution.height() / 2.0,
            );
            if mouse_keys.just_pressed(MouseButton::Left) {
                wind.cursor.visible = false;
                wind.cursor.grab_mode = CursorGrabMode::Confined;
                wind.set_cursor_position(Some(pos));
                mouse_events.clear();
            } else if mouse_keys.just_released(MouseButton::Left) {
                wind.cursor.visible = true;
                wind.cursor.grab_mode = CursorGrabMode::None;
            } else if mouse_keys.pressed(MouseButton::Left) {
                wind.set_cursor_position(Some(pos));
            } else {
                return;
            }
        }

        let mut mouse_delta = Vec2::ZERO;
        for e in mouse_events.read() {
            mouse_delta += e.delta;
        }
        mouse_delta *= 0.002;

        {
            let mut rotation = transform.rotation;
            let yrotation = Quat::from_axis_angle(Vec3::Y, -mouse_delta.x);
            let xrotation = Quat::from_axis_angle(transform.local_x(), -mouse_delta.y);

            rotation = xrotation * rotation;
            let is_upside_down = rotation.mul_vec3(Vec3::Y).y < 0.0;
            if is_upside_down {
                rotation = transform.rotation;
            }

            rotation = yrotation * rotation;
            transform.rotation = rotation;
        }

        let speed = 0.2;
        let forward = transform.forward();
        let right = transform.local_x();
        let mut translation = Vec3::ZERO;

        if keys.pressed(KeyCode::W) {
            translation += forward * speed;
        }
        if keys.pressed(KeyCode::S) {
            translation -= forward * speed;
        }
        if keys.pressed(KeyCode::A) {
            translation -= right * speed;
        }
        if keys.pressed(KeyCode::D) {
            translation += right * speed;
        }
        if keys.pressed(KeyCode::ShiftLeft) {
            translation *= 5.0;
        }
        if keys.pressed(KeyCode::ControlLeft) {
            translation /= 5.0;
        }
        transform.translation += translation;
    }
}

mod spectator {
    use super::*;

    #[derive(Component)]
    pub struct SpectatorEntity;

    pub struct Spectator;
    impl Plugin for Spectator {
        fn build(&self, app: &mut App) {
            app.add_systems(Startup, setup)
                .add_systems(Update, switch_mode)
                .add_systems(Update, update.run_if(in_state(ControlsState::Spectator)));
        }
    }

    fn switch_mode(
        mut keys: ResMut<Input<KeyCode>>,
        mut spectator: Query<(&mut Camera, &mut SpectatorEntity)>,
        curr_state: Res<State<ControlsState>>,
        mut state: ResMut<NextState<ControlsState>>,
    ) {
        if !keys.just_pressed(KeyCode::Key1) {
            return;
        }

        let (mut camera, _) = spectator.single_mut();

        if *curr_state.get() == ControlsState::Spectator {
            state.set(ControlsState::Player);
            camera.is_active = false;
        } else {
            state.set(ControlsState::Spectator);
            camera.is_active = true;
        }

        // dosen't fix the bug T-T
        // keys.press(KeyCode::Escape);
    }

    fn setup(mut commands: Commands, mut _meshes: ResMut<Assets<Mesh>>) {
        let transform = Transform::from_xyz(5.0, 5.0, -5.0).looking_at(Vec3::ZERO, Vec3::Y);
        commands.spawn((
            Camera3dBundle {
                transform,
                camera: Camera {
                    order: 1,
                    is_active: false,
                    ..Default::default()
                },
                ..default()
            },
            SpectatorEntity,
        ));
    }

    fn update(
        mut spectator: Query<(&mut Transform, &mut SpectatorEntity)>,
        keys: Res<Input<KeyCode>>,
        mouse_keys: Res<Input<MouseButton>>,
        mut mouse_events: EventReader<MouseMotion>,
        // time: Res<Time>,
        mut window: Query<&mut Window, With<PrimaryWindow>>,
    ) {
        let mut wind = window.single_mut();
        if keys.just_pressed(KeyCode::Escape) {
            wind.cursor.grab_mode = match wind.cursor.grab_mode {
                CursorGrabMode::Locked => {
                    wind.cursor.visible = true;
                    CursorGrabMode::None
                }
                _ => {
                    wind.cursor.visible = false;
                    CursorGrabMode::Locked
                }
            };
            let pos = Vec2::new(
                wind.resolution.width() / 2.0,
                wind.resolution.height() / 2.0,
            );
            wind.set_cursor_position(Some(pos));
            mouse_events.clear();
        }

        let (mut transform, _) = spectator.single_mut();
        if wind.cursor.grab_mode != CursorGrabMode::Locked {
            let pos = Vec2::new(
                wind.resolution.width() / 2.0,
                wind.resolution.height() / 2.0,
            );
            if mouse_keys.just_pressed(MouseButton::Left) {
                wind.cursor.visible = false;
                wind.cursor.grab_mode = CursorGrabMode::Confined;
                wind.set_cursor_position(Some(pos));
                mouse_events.clear();
            } else if mouse_keys.just_released(MouseButton::Left) {
                wind.cursor.visible = true;
                wind.cursor.grab_mode = CursorGrabMode::None;
            } else if mouse_keys.pressed(MouseButton::Left) {
                wind.set_cursor_position(Some(pos));
            } else {
                return;
            }
        }

        let mut mouse_delta = Vec2::ZERO;
        for e in mouse_events.read() {
            mouse_delta += e.delta;
        }
        mouse_delta *= 0.002;

        {
            let mut rotation = transform.rotation;
            let yrotation = Quat::from_axis_angle(Vec3::Y, -mouse_delta.x);
            let xrotation = Quat::from_axis_angle(transform.local_x(), -mouse_delta.y);

            rotation = xrotation * rotation;
            let is_upside_down = rotation.mul_vec3(Vec3::Y).y < 0.0;
            if is_upside_down {
                rotation = transform.rotation;
            }

            rotation = yrotation * rotation;
            transform.rotation = rotation;
        }

        let speed = 0.2;
        let forward = transform.forward();
        let right = transform.local_x();
        let mut translation = Vec3::ZERO;

        if keys.pressed(KeyCode::W) {
            translation += forward * speed;
        }
        if keys.pressed(KeyCode::S) {
            translation -= forward * speed;
        }
        if keys.pressed(KeyCode::A) {
            translation -= right * speed;
        }
        if keys.pressed(KeyCode::D) {
            translation += right * speed;
        }
        if keys.pressed(KeyCode::ShiftLeft) {
            translation *= 5.0;
        }
        if keys.pressed(KeyCode::ControlLeft) {
            translation /= 5.0;
        }
        transform.translation += translation;
    }
}

fn lerp(i: Vec3, f: Vec3, s: f32, dt: f32) -> Vec3 {
    let s = (1.0 - s).powf(dt * 120.0);
    i * s + f * (1.0 - s)
}

// https://youtu.be/ibkT5ao8kGY
fn slerp(i: Vec3, f: Vec3, s: f32, dt: f32) -> Vec3 {
    let s = (1.0 - s).powf(dt * 120.0);
    let theta = i.dot(f).acos();
    if theta.sin() == 0.0 {
        return i + Vec3::splat(0.00000001);
    }
    ((s * theta).sin() / theta.sin()) * i + (((1.0 - s) * theta).sin() / theta.sin()) * f
}

/// set up a simple 3D scene
fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<CustomMaterial>>,
    mut std_materials: ResMut<Assets<StandardMaterial>>,
    // mut depth_materials: ResMut<Assets<PrepassOutputMaterial>>,
    asset_server: Res<AssetServer>,
) {
    // camera
    // commands.spawn((
    //     Camera3dBundle {
    //         transform: Transform::from_xyz(-2.0, 3., 5.0).looking_at(Vec3::ZERO, Vec3::Y),
    //         ..default()
    //     },
    //     // To enable the prepass you need to add the components associated with the ones you need
    //     // This will write the depth buffer to a texture that you can use in the main pass
    //     DepthPrepass,
    //     // This will generate a texture containing world normals (with normal maps applied)
    //     NormalPrepass,
    //     // This will generate a texture containing screen space pixel motion vectors
    //     MotionVectorPrepass,
    // ));

    // plane
    commands.spawn(PbrBundle {
        mesh: meshes.add(shape::Plane::from_size(5.0).into()),
        material: std_materials.add(Color::rgb(0.3, 0.5, 0.3).into()),
        ..default()
    });

    // A quad that shows the outputs of the prepass
    // To make it easy, we just draw a big quad right in front of the camera.
    // For a real application, this isn't ideal.
    // commands.spawn((
    //     MaterialMeshBundle {
    //         mesh: meshes.add(shape::Quad::new(Vec2::new(20.0, 20.0)).into()),
    //         material: depth_materials.add(PrepassOutputMaterial {
    //             settings: ShowPrepassSettings::default(),
    //         }),
    //         transform: Transform::from_xyz(-0.75, 1.25, 3.0)
    //             .looking_at(Vec3::new(2.0, -2.5, -5.0), Vec3::Y),
    //         ..default()
    //     },
    //     NotShadowCaster,
    // ));

    // Opaque cube
    // commands.spawn((
    //     MaterialMeshBundle {
    //         mesh: meshes.add(Mesh::from(shape::Cube { size: 1.0 })),
    //         material: materials.add(CustomMaterial {
    //             color: Color::WHITE,
    //             color_texture: None,
    //             alpha_mode: AlphaMode::Opaque,
    //         }),
    //         transform: Transform::from_xyz(-1.0, 0.5, 0.0),
    //         ..default()
    //     },
    //     Rotates,
    // ));

    // // Cube with alpha mask
    // commands.spawn(PbrBundle {
    //     mesh: meshes.add(Mesh::from(shape::Cube { size: 1.0 })),
    //     material: std_materials.add(StandardMaterial {
    //         alpha_mode: AlphaMode::Mask(1.0),
    //         base_color_texture: Some(asset_server.load("icons/icon.png")),
    //         ..default()
    //     }),
    //     transform: Transform::from_xyz(0.0, 0.5, 0.0),
    //     ..default()
    // });

    // // Cube with alpha blending.
    // // Transparent materials are ignored by the prepass
    // commands.spawn(MaterialMeshBundle {
    //     mesh: meshes.add(Mesh::from(shape::Cube { size: 1.0 })),
    //     material: materials.add(CustomMaterial {
    //         color: Color::WHITE,
    //         color_texture: Some(asset_server.load("icons/icon.png")),
    //         alpha_mode: AlphaMode::Blend,
    //     }),
    //     transform: Transform::from_xyz(1.0, 0.5, 0.0),
    //     ..default()
    // });

    // light
    // commands.spawn(PointLightBundle {
    //     point_light: PointLight {
    //         intensity: 1500.0,
    //         shadows_enabled: true,
    //         ..default()
    //     },
    //     transform: Transform::from_xyz(4.0, 8.0, 4.0),
    //     ..default()
    // });

    // let style = TextStyle {
    //     font_size: 18.0,
    //     ..default()
    // };

    // commands.spawn(
    //     TextBundle::from_sections(vec![
    //         TextSection::new("Prepass Output: transparent\n", style.clone()),
    //         TextSection::new("\n\n", style.clone()),
    //         TextSection::new("Controls\n", style.clone()),
    //         TextSection::new("---------------\n", style.clone()),
    //         TextSection::new("Space - Change output\n", style),
    //     ])
    //     .with_style(Style {
    //         position_type: PositionType::Absolute,
    //         top: Val::Px(10.0),
    //         left: Val::Px(10.0),
    //         ..default()
    //     }),
    // );
}

// This is the struct that will be passed to your shader
#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub struct CustomMaterial {
    #[uniform(0)]
    color: Color,
    #[texture(1)]
    #[sampler(2)]
    color_texture: Option<Handle<Image>>,
    alpha_mode: AlphaMode,
}

/// Not shown in this example, but if you need to specialize your material, the specialize
/// function will also be used by the prepass
impl Material for CustomMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/custom_material.wgsl".into()
    }

    fn alpha_mode(&self) -> AlphaMode {
        self.alpha_mode
    }

    // You can override the default shaders used in the prepass if your material does
    // anything not supported by the default prepass
    // fn prepass_fragment_shader() -> ShaderRef {
    //     "shaders/custom_material.wgsl".into()
    // }
}

#[derive(Component)]
struct Rotates;

fn rotate(mut q: Query<&mut Transform, With<Rotates>>, time: Res<Time>) {
    for mut t in q.iter_mut() {
        let rot = (time.elapsed_seconds().sin() * 0.5 + 0.5) * std::f32::consts::PI * 2.0;
        t.rotation = Quat::from_rotation_z(rot);
    }
}

// #[derive(Debug, Clone, Default, ShaderType)]
// struct ShowPrepassSettings {
//     show_depth: u32,
//     show_normals: u32,
//     show_motion_vectors: u32,
//     padding_1: u32,
//     padding_2: u32,
// }

// // This shader simply loads the prepass texture and outputs it directly
// #[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
// pub struct PrepassOutputMaterial {
//     #[uniform(0)]
//     settings: ShowPrepassSettings,
// }

// impl Material for PrepassOutputMaterial {
//     fn fragment_shader() -> ShaderRef {
//         "shaders/show_prepass.wgsl".into()
//     }

//     // This needs to be transparent in order to show the scene behind the mesh
//     fn alpha_mode(&self) -> AlphaMode {
//         AlphaMode::Blend
//     }
// }

// Every time you press space, it will cycle between transparent, depth and normals view
// fn toggle_prepass_view(
//     mut prepass_view: Local<u32>,
//     keycode: Res<Input<KeyCode>>,
//     material_handle: Query<&Handle<PrepassOutputMaterial>>,
//     mut materials: ResMut<Assets<PrepassOutputMaterial>>,
//     mut text: Query<&mut Text>,
// ) {
//     if keycode.just_pressed(KeyCode::Space) {
//         *prepass_view = (*prepass_view + 1) % 4;

//         let label = match *prepass_view {
//             0 => "transparent",
//             1 => "depth",
//             2 => "normals",
//             3 => "motion vectors",
//             _ => unreachable!(),
//         };
//         let mut text = text.single_mut();
//         text.sections[0].value = format!("Prepass Output: {label}\n");
//         for section in &mut text.sections {
//             section.style.color = Color::WHITE;
//         }

//         let handle = material_handle.single();
//         let mat = materials.get_mut(handle).unwrap();
//         mat.settings.show_depth = (*prepass_view == 1) as u32;
//         mat.settings.show_normals = (*prepass_view == 2) as u32;
//         mat.settings.show_motion_vectors = (*prepass_view == 3) as u32;
//     }
// }
