//! Bevy has an optional prepass that is controlled per-material. A prepass is a rendering pass that runs before the main pass.
//! It will optionally generate various view textures. Currently it supports depth, normal, and motion vector textures.
//! The textures are not generated for any material using alpha blending.

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
        MaterialPlugin::<ChunkMaterial>::default(),
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
    .add_plugins(player::Player)
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
    use bevy::{
        app::{Plugin, Startup},
        asset::{AssetServer, Assets, Handle, load_internal_asset},
        core_pipeline::{
            clear_color::ClearColorConfig,
            core_3d::{
                graph::input::VIEW_ENTITY, Camera3d, Camera3dBundle, Transparent3d, CORE_3D,
            },
            tonemapping::Tonemapping,
        },
        ecs::{
            bundle::Bundle,
            component::Component,
            entity::Entity,
            query::{ROQueryItem, With},
            schedule::{IntoSystemConfigs, State, States, NextState, OnEnter},
            system::{
                lifetimeless::{Read, SQuery, SRes},
                Commands, Query, Res, ResMut, Resource, RunSystemOnce, SystemParamItem,
            },
            world::FromWorld,
        },
        pbr::{
            DrawMesh, MeshPipeline, MeshPipelineKey, MeshPipelineViewLayoutKey, MeshTransforms,
            SetMeshBindGroup, SetMeshViewBindGroup, MaterialMeshBundle, RenderMeshInstances, RenderMeshInstance, MaterialBindGroupId, MeshFlags,
        },
        render::{
            camera::{Camera, CameraRenderGraph, Projection, RenderTarget, OrthographicProjection, ScalingMode},
            main_graph::node::CAMERA_DRIVER,
            mesh::{Mesh, MeshVertexBufferLayout, shape::{Cube, UVSphere}, GpuBufferInfo},
            primitives::{Frustum, Sphere},
            render_asset::RenderAssets,
            render_graph::{RenderGraph, SlotInfo, SlotType},
            render_phase::{
                AddRenderCommand, DrawFunctions, PhaseItem, RenderCommand, RenderCommandResult,
                RenderPhase, SetItemPipeline, TrackedRenderPass,
            },
            render_resource::{
                AsBindGroup, BindGroup, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
                BindGroupLayoutEntry, BindingResource,
                BindingType::{self, StorageTexture},
                BufferBindingType, Extent3d, PipelineCache, PrimitiveTopology,
                RenderPipelineDescriptor, Shader, ShaderStages, SpecializedMeshPipeline,
                SpecializedMeshPipelineError, SpecializedMeshPipelines, StorageTextureAccess,
                Texture, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages,
                TextureView, TextureViewDescriptor, TextureViewDimension,
            },
            renderer::{RenderDevice, RenderQueue},
            texture::Image,
            view::{ExtractedView, VisibleEntities, Visibility, InheritedVisibility, ViewVisibility, NoFrustumCulling},
            Render, RenderApp, RenderSet, extract_component::{ExtractComponentPlugin, ExtractComponent}, extract_resource::{ExtractResource, ExtractResourcePlugin}, ExtractSchedule, Extract,
        },
        transform::components::{GlobalTransform, Transform}, core::Name, math::Vec3,
    };

    pub struct RenderPlugin;

    impl Plugin for RenderPlugin {
        fn build(&self, app: &mut bevy::prelude::App) {
            app.add_plugins(WorldPlugin).add_plugins(VoxelizationPlugin);
            app.add_systems(Startup, test_plugin);
        }

        fn finish(&self, app: &mut bevy::prelude::App) {
            // let mut smox_graph = RenderGraph::default();

            // let input_node_id =
            //     smox_graph.set_input(vec![SlotInfo::new(VIEW_ENTITY, SlotType::Entity)]);

            // render_graph.add_node_edge(input_node_id, CAMERA_DRIVER);

            // let mut render_graph = render_app.world.resource_mut::<RenderGraph>();
            // smox_graph.add_slot_edge(input_node_id, VIEW_ENTITY, CAMERA_DRIVER, "view");
            // smox_graph.add_node(CAMERA_DRIVER, CAMERA_DRIVER);

            // render_graph.add_sub_graph("smox", smox_graph);
        }
    }

    fn test_plugin(mut commands: Commands, mut meshes: ResMut<Assets<Mesh>>) {
        commands.spawn((
            Name::new("voxelization test mesh"),
            VoxelizeBundle {
                mesh: meshes.add(Mesh::from(UVSphere::default())),
                transform: Transform::from_translation(Vec3::ZERO),
                ..Default::default()
            },
        ));
    }

    #[derive(Bundle)]
    pub struct VoxCamBundle {
        pub camera_render_graph: CameraRenderGraph,
        pub camera: Camera,
        pub camera_3d: Camera3d,
        pub projection: Projection,
        pub visible_entities: VisibleEntities,
        pub frustun: Frustum,
        pub global_transform: GlobalTransform,
        pub transform: Transform,
        pub tonemapping: Tonemapping,
    }
    impl Default for VoxCamBundle {
        fn default() -> Self {
            Self {
                camera_render_graph: CameraRenderGraph::new(CORE_3D),
                // camera_render_graph: CameraRenderGraph::new("smox"),
                tonemapping: Tonemapping::default(), // TODO: look into this
                camera: Default::default(),
                camera_3d: Default::default(),
                projection: Default::default(),
                visible_entities: Default::default(),
                frustun: Default::default(),
                global_transform: Default::default(),
                transform: Default::default(),
            }
        }
    }

    struct WorldPlugin;
    impl Plugin for WorldPlugin {
        fn build(&self, app: &mut bevy::prelude::App) {
        }

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

            // TODO: can use AsBindGroup to generate this boilerplate
            let world_data_bind_group_layout =
                render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: Some("world data bind group"),
                    entries: &[BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::FRAGMENT | ShaderStages::COMPUTE,
                        ty: BindingType::StorageTexture {
                            access: StorageTextureAccess::ReadWrite,
                            format: TextureFormat::R8Uint,
                            view_dimension: TextureViewDimension::D3,
                        },
                        count: None,
                    }],
                });
            let world_data_bind_group = render_device.create_bind_group(
                None,
                &world_data_bind_group_layout,
                &[BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&transient_world_texture_view),
                }],
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
            let mut images = app.world.resource_mut::<Assets<Image>>();
            let transient_world_camera_target = images.add(transient_world_camera_target_image);

            app.add_plugins(ExtractResourcePlugin::<WorldData>::default());
            let world_data = WorldData {
                bind_group_layout: world_data_bind_group_layout,
                bind_group: world_data_bind_group,
                transient_world_texture,
                transient_world_texture_view,
                transient_world_camera_target,
            };
            let wd2 = WorldData::extract_resource(&world_data);
            app.insert_resource(world_data);
            let render_app = app.sub_app_mut(RenderApp);
            render_app.insert_resource(wd2);
            render_app.add_systems(ExtractSchedule, extract_meshes);
        }
    }

    fn extract_meshes(mut commands: Commands, mut instances: ResMut<RenderMeshInstances>, meshes: Extract<Query<(Entity, &Voxelize, &Handle<Mesh>, &GlobalTransform)>>) {
        for (e, v, h, t) in meshes.iter() {
            commands.spawn((
                Voxelize,
                h.clone_weak(),
                t.clone(),
            ));
            let mut flags = MeshFlags::empty();
            if t.affine().matrix3.determinant().is_sign_positive() {
                flags |= MeshFlags::SIGN_DETERMINANT_MODEL_3X3;
            }
            instances.extend([(e, RenderMeshInstance {
                transforms: MeshTransforms { transform: (&t.affine()).into(), previous_transform: (&t.affine()).into(), flags: flags.bits() },
                mesh_asset_id: h.id(),
                material_bind_group_id: MaterialBindGroupId::default(),
                shadow_caster: false,
                automatic_batching: false,
            })]);
        }
    }

    #[derive(Resource, Clone, ExtractResource)]
    struct WorldData {
        bind_group_layout: BindGroupLayout,
        bind_group: BindGroup,

        transient_world_texture: Texture,
        transient_world_texture_view: TextureView,
        transient_world_camera_target: Handle<Image>,
    }

    #[derive(Component)]
    struct VoxelizationCam;

    // slap this on meshes you want voxelized into the transient world
    #[derive(Component, Default, Clone, ExtractComponent)]
    struct Voxelize;

    #[derive(Bundle, Default)]
    pub struct VoxelizeBundle {
        mesh: Handle<Mesh>,
        transform: Transform,
        global_transform: GlobalTransform,
        visibility: Visibility,
        inherited_visibility: InheritedVisibility,
        voxelize: Voxelize,
        view_vis: ViewVisibility,
        nocull: NoFrustumCulling,
    }

    const VOXELIZATION_SHADER_HANDLE: Handle<Shader> =
        Handle::weak_from_u128(1975691635883203525);
    struct VoxelizationPlugin;
    impl Plugin for VoxelizationPlugin {
        fn build(&self, app: &mut bevy::prelude::App) {
            load_internal_asset!(
                app,
                VOXELIZATION_SHADER_HANDLE,
                "../assets/shaders/voxelize.wgsl",
                Shader::from_wgsl
            );

            app.add_systems(Startup, setup_cameras);
        }
        fn finish(&self, app: &mut bevy::prelude::App) {
            let render_app = app.sub_app_mut(RenderApp);
            render_app
                .add_render_command::<Transparent3d, CustomDraw>()
                .init_resource::<VoxelizationPipeline>()
                .init_resource::<SpecializedMeshPipelines<VoxelizationPipeline>>();
            render_app.add_systems(
                Render,
                queue_for_voxelization.in_set(RenderSet::QueueMeshes),
            );
        }
    }

    fn setup_cameras(mut commands: Commands, world_data: Res<WorldData>) {
        // 16 voxels per unit
        let side = (100.0 / 16.0)/2.0;

        let transform = Transform::from_translation(Vec3::ZERO);
        for i in -3..0 {
            let transform = match i {
                -3 => transform.looking_to(Vec3::X, Vec3::Y),
                -2 => transform.looking_to(Vec3::Y, Vec3::Z),
                -1 => transform.looking_to(Vec3::Z, Vec3::Y),
                _ => unreachable!(),
            };
            commands.spawn((
                // TODO: look into using a different CameraRenderGraph for voxelization
                Camera3dBundle {
                    camera: Camera {
                        order: i,
                        target: RenderTarget::Image(world_data.transient_world_camera_target.clone()),
                        ..Default::default()
                    },
                    camera_3d: Camera3d {
                        clear_color: ClearColorConfig::None,
                        ..Default::default()
                    },
                    projection: Projection::Orthographic(OrthographicProjection {
                        near: -side,
                        far: side,
                        scaling_mode: ScalingMode::Fixed {
                            width: 2.0*side,
                            height: 2.0*side,
                        },
                        ..Default::default()
                    }),
                    transform,
                    ..Default::default()
                },
                VoxelizationCam,
                Name::new("voxelization cam"),
            ));
        }
    }

    // https://github.com/bevyengine/bevy/blob/22e39c4abf6e2fdf99ba0820b3c35db73be71347/examples/2d/mesh2d_manual.rs#L338
    #[allow(clippy::too_many_arguments)]
    fn queue_for_voxelization(
        transparent_draw_functions: Res<DrawFunctions<Transparent3d>>,
        pipeline: Res<VoxelizationPipeline>,
        mut pipelines: ResMut<SpecializedMeshPipelines<VoxelizationPipeline>>,
        pipeline_cache: Res<PipelineCache>,
        render_meshes: Res<RenderAssets<Mesh>>,
        mesh_query: Query<(Entity, &GlobalTransform, &Handle<Mesh>), With<Voxelize>>,
        mut test_query: Query<(Entity, &Handle<Mesh>)>,
        mut views: Query<(&mut RenderPhase<Transparent3d>, &ExtractedView)>,
    ) {
        for e in test_query.iter() {
            // dbg!(e);
        }

        let draw_function = transparent_draw_functions
            .read()
            .get_id::<CustomDraw>()
            .unwrap();
        let key = MeshPipelineKey::from_primitive_topology(PrimitiveTopology::TriangleList);

        // a camera is a view
        for (mut phase, view) in views.iter_mut() {
            let rangefinder = view.rangefinder3d();
            for (entity, transforms, mesh_handle) in mesh_query.iter() {
                let Some(mesh) = render_meshes.get(mesh_handle) else {
                    continue;
                };
                // dbg!(entity, rangefinder.distance_translation(&transforms.translation()));

                let pipeline = pipelines
                    .specialize(&pipeline_cache, &pipeline, key, &mesh.layout)
                    .unwrap();
                phase.add(Transparent3d {
                    entity,
                    pipeline,
                    distance: rangefinder.distance_translation(&transforms.translation()),
                    draw_function,
                    batch_range: 0..1,
                    dynamic_offset: None,
                });
            }
        }
    }

    #[derive(Resource)]
    struct VoxelizationPipeline {
        vertex_shader: Handle<Shader>,
        fragment_shader: Handle<Shader>,
        mesh_pipeline: MeshPipeline,
        world_bind_group_layout: BindGroupLayout,
        // voxelization_bind_group_layout: BindGroupLayout,
    }
    impl FromWorld for VoxelizationPipeline {
        fn from_world(world: &mut bevy::prelude::World) -> Self {
            let asset_server = world.resource::<AssetServer>();
            // let render_device = world.resource::<RenderDevice>();
            let world_data = world.resource::<WorldData>();
            Self {
                mesh_pipeline: world.resource::<MeshPipeline>().clone(),
                vertex_shader: asset_server.load("shaders/voxelize.wgsl"),
                fragment_shader: asset_server.load("shaders/voxelize.wgsl"),
                world_bind_group_layout: world_data.bind_group_layout.clone(),
                // voxelization_bind_group_layout: render_device.create_bind_group_layout(
                //     &BindGroupLayoutDescriptor {
                //         label: None,
                //         entries: &[],
                //     },
                // ),
            }
        }
    }
    impl SpecializedMeshPipeline for VoxelizationPipeline {
        type Key = MeshPipelineKey;

        fn specialize(
            &self,
            key: Self::Key,
            layout: &MeshVertexBufferLayout,
        ) -> Result<RenderPipelineDescriptor, SpecializedMeshPipelineError> {
            let mut desc = self.mesh_pipeline.specialize(key, layout)?;
            desc.vertex.shader = self.vertex_shader.clone();
            desc.fragment.as_mut().unwrap().shader = self.fragment_shader.clone();
            // desc.vertex.shader = VOXELIZATION_SHADER_HANDLE;
            // desc.fragment.as_mut().unwrap().shader = VOXELIZATION_SHADER_HANDLE;
            desc.layout = vec![
                self.mesh_pipeline
                    .get_view_layout(MeshPipelineViewLayoutKey::NORMAL_PREPASS)
                    .clone(),
                self.mesh_pipeline.mesh_layouts.morphed_skinned.clone(),
                self.world_bind_group_layout.clone(),
                // self.voxelization_bind_group_layout.clone(),
            ];
            desc.primitive.cull_mode = None;
            Ok(desc)
        }
    }

    type CustomDraw = (
        SetItemPipeline,
        SetMeshViewBindGroup<0>,
        SetMeshBindGroup<1>,
        SetWorldBindGroup<2>,
        // SetVoxelizationBindGroup<3>,
        // DrawMesh,
        DrawMeAMesh,
    );
    pub struct DrawMeAMesh;
    impl<P: PhaseItem> RenderCommand<P> for DrawMeAMesh {
        type Param = (SRes<RenderAssets<Mesh>>, SRes<RenderMeshInstances>);
        type ViewWorldQuery = ();
        type ItemWorldQuery = ();
        #[inline]
        fn render<'w>(
            item: &P,
            _view: (),
            _item_query: (),
            (meshes, mesh_instances): SystemParamItem<'w, '_, Self::Param>,
            pass: &mut TrackedRenderPass<'w>,
        ) -> RenderCommandResult {
            let meshes = meshes.into_inner();
            let mesh_instances = mesh_instances.into_inner();

            dbg!();
            let Some(mesh_instance) = mesh_instances.get(&item.entity()) else {
                return RenderCommandResult::Failure;
            };
            dbg!();
            let Some(gpu_mesh) = meshes.get(mesh_instance.mesh_asset_id) else {
                return RenderCommandResult::Failure;
            };

            pass.set_vertex_buffer(0, gpu_mesh.vertex_buffer.slice(..));

            let batch_range = item.batch_range();
            dbg!(&batch_range);
            match &gpu_mesh.buffer_info {
                GpuBufferInfo::Indexed {
                    buffer,
                    index_format,
                    count,
                } => {
                    dbg!();
                    pass.set_index_buffer(buffer.slice(..), 0, *index_format);
                    pass.draw_indexed(0..*count, 0, batch_range.clone());
                }
                GpuBufferInfo::NonIndexed => {
                    dbg!();
                    pass.draw(0..gpu_mesh.vertex_count, batch_range.clone());
                }
            }
            RenderCommandResult::Success
        }
    }
    // we only have 1 world, so we use a resource
    // and set bind group for this one world
    struct SetWorldBindGroup<const I: usize>;
    impl<P: PhaseItem, const I: usize> RenderCommand<P> for SetWorldBindGroup<I> {
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

            pass.set_bind_group(I, &world.bind_group, &[]);

            RenderCommandResult::Success
        }
    }

    // // we might have many things we want to voxelize, so we use a component
    // // we set this bind group when rendering any object with this component
    // #[derive(Component)]
    // struct VoxelizationBindGroup(BindGroup);
    // struct SetVoxelizationBindGroup<const I: usize>;
    // // execute this render command for ever item P with VoxelizationBindGroup component
    // impl<P: PhaseItem, const I: usize> RenderCommand<P> for SetVoxelizationBindGroup<I> {
    //     type Param = SQuery<Read<VoxelizationBindGroup>>;

    //     type ViewWorldQuery = ();

    //     type ItemWorldQuery = ();

    //     fn render<'w>(
    //         item: &P,
    //         view: ROQueryItem<'w, Self::ViewWorldQuery>,
    //         entity: ROQueryItem<'w, Self::ItemWorldQuery>,
    //         param: SystemParamItem<'w, '_, Self::Param>,
    //         pass: &mut TrackedRenderPass<'w>,
    //     ) -> RenderCommandResult {
    //         // TODO: find out what this 'inner' stuff is
    //         let vox_bind_group = param.get_inner(item.entity()).unwrap();

    //         pass.set_bind_group(I, &vox_bind_group.0, &[]);

    //         RenderCommandResult::Success
    //     }
    // }
}

mod chunk {
    use bevy::pbr::DefaultOpaqueRendererMethod;
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
                    update_chunk_material,
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
                player_position: Vec3::ZERO,
                resolution: Vec2::ZERO,
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
            assert_eq!(self.side%2, 0, "side should be divisible by 2");
            let new_side = self.side/2;
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
                            voxel(z*2 + 0, y*2 + 0, x*2 + 0),
                            voxel(z*2 + 0, y*2 + 0, x*2 + 1),
                            voxel(z*2 + 0, y*2 + 1, x*2 + 0),
                            voxel(z*2 + 0, y*2 + 1, x*2 + 1),
                            voxel(z*2 + 1, y*2 + 0, x*2 + 0),
                            voxel(z*2 + 1, y*2 + 0, x*2 + 1),
                            voxel(z*2 + 1, y*2 + 1, x*2 + 0),
                            voxel(z*2 + 1, y*2 + 1, x*2 + 1),
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
    #[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
    pub struct ChunkMaterial {
        #[uniform(0)]
        pub side: u32,
        #[texture(2, dimension = "1d", sample_type = "float", filterable = false)]
        pub materials: Handle<Image>,
        #[uniform(3)]
        pub player_position: Vec3,
        #[uniform(4)]
        pub resolution: Vec2,
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

        commands.spawn(DirectionalLightBundle {
            directional_light: DirectionalLight {
                illuminance: 3000.0,
                shadows_enabled: true,
                ..Default::default()
            },
            transform: Transform::from_xyz(1.0, 1.0, 1.0)
                .with_rotation(Quat::from_axis_angle(Vec3::X, 3.5)),
            ..Default::default()
        });
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

    // TODO: use a custom pipeline and remove this from this material
    // or just use ExtractComponentPlugin<{ pos: Vec3 }>
    fn update_chunk_material(
        players: Query<(&PlayerEntity, &Transform)>,
        mut chunk_materials: ResMut<Assets<ChunkMaterial>>,
        windows: Query<&Window>,
        chunks: Query<(&Handle<ChunkMaterial>, &GlobalTransform)>,
    ) {
        let (_, player) = players.single();
        let window = windows.single();
        let width = window.resolution.physical_width() as _;
        let height = window.resolution.physical_height() as _;
        for material in chunk_materials.iter_mut() {
            material.1.player_position = player.translation;
            material.1.resolution = Vec2::new(width, height);
        }

        for (material, pos) in chunks.iter() {
            chunk_materials.get_mut(material).unwrap().chunk_position = pos.translation();
        }
    }
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
                player_position: Vec3::ZERO,
                resolution: Vec2::ZERO,
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
    use crate::render::VoxCamBundle;

    use super::*;

    #[derive(Component)]
    pub struct PlayerEntity;

    pub struct Player;
    impl Plugin for Player {
        fn build(&self, app: &mut App) {
            app.add_systems(Startup, setup)
                .add_systems(Update, update.run_if(in_state(ControlsState::Player)));
        }
    }

    fn setup(mut commands: Commands, mut _meshes: ResMut<Assets<Mesh>>) {
        let transform = Transform::from_xyz(5.0, 5.0, -5.0).looking_at(Vec3::ZERO, Vec3::Y);
        commands.spawn((
            VoxCamBundle {
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
    commands.spawn((
        MaterialMeshBundle {
            mesh: meshes.add(Mesh::from(shape::Cube { size: 1.0 })),
            material: materials.add(CustomMaterial {
                color: Color::WHITE,
                color_texture: None,
                alpha_mode: AlphaMode::Opaque,
            }),
            transform: Transform::from_xyz(-1.0, 0.5, 0.0),
            ..default()
        },
        Rotates,
    ));

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
    commands.spawn(PointLightBundle {
        point_light: PointLight {
            intensity: 1500.0,
            shadows_enabled: true,
            ..default()
        },
        transform: Transform::from_xyz(4.0, 8.0, 4.0),
        ..default()
    });

    let style = TextStyle {
        font_size: 18.0,
        ..default()
    };

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
