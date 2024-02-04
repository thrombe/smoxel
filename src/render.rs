use core::panic;
use std::{cmp::Reverse, ops::Range};

use bevy::{
    app::{Plugin, Startup},
    asset::{Asset, AssetApp, AssetEvent, AssetId, AssetServer, Assets, Handle},
    core::Name,
    core_pipeline::{
        core_3d::{
            graph::node::{MAIN_OPAQUE_PASS, MAIN_TRANSPARENT_PASS, PREPASS},
            Camera3d, Camera3dBundle, CORE_3D,
        },
        prepass::{DepthPrepass, NormalPrepass},
        tonemapping::Tonemapping,
    },
    ecs::{
        bundle::Bundle,
        component::Component,
        entity::Entity,
        event::EventReader,
        query::{Has, QueryItem, ROQueryItem, With},
        schedule::IntoSystemConfigs,
        system::{
            lifetimeless::SRes, Commands, Local, Query, Res, ResMut, Resource, SystemParamItem,
        },
        world::FromWorld,
    },
    math::{UVec2, Vec3, Vec4},
    pbr::{
        AlphaMode, DrawMesh, Material, MaterialMeshBundle, MaterialProperties, MeshPipeline,
        MeshPipelineKey, OpaqueRendererMethod, PreparedMaterial, RenderMaterialInstances,
        RenderMaterials, RenderMeshInstances, SetMaterialBindGroup, SetMeshBindGroup,
        SetMeshViewBindGroup,
    },
    reflect::TypePath,
    render::{
        batching::batch_and_prepare_render_phase,
        camera::{
            Camera, ExtractedCamera, OrthographicProjection, Projection, ScalingMode,
            TemporalJitter,
        },
        color::Color,
        extract_component::ExtractComponent,
        extract_instances::ExtractInstancesPlugin,
        main_graph::node::CAMERA_DRIVER,
        mesh::{shape, Mesh, MeshVertexBufferLayout},
        render_asset::{prepare_assets, RenderAssets},
        render_graph::{
            Node, NodeRunError, RenderGraph, RenderGraphApp, RenderGraphContext, ViewNode,
            ViewNodeRunner,
        },
        render_phase::{
            sort_phase_system, AddRenderCommand, CachedRenderPipelinePhaseItem, DrawFunctionId,
            DrawFunctions, PhaseItem, RenderCommand, RenderCommandResult, RenderPhase,
            SetItemPipeline, TrackedRenderPass,
        },
        render_resource::{
            AsBindGroup, AsBindGroupError, BindGroup, BindGroupEntry, BindGroupLayout,
            BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingResource,
            BindingType::{self},
            BufferBindingType, BufferSize, CachedComputePipelineId, CachedRenderPipelineId,
            ColorTargetState, ColorWrites, ComputePassDescriptor, ComputePipeline,
            ComputePipelineDescriptor, ComputePipelineId, Extent3d, Face, ImageSubresourceRange,
            LoadOp, Operations, PipelineCache, RenderPassColorAttachment,
            RenderPassDepthStencilAttachment, RenderPassDescriptor, RenderPipelineDescriptor,
            Shader, ShaderDefVal, ShaderRef, ShaderSize, ShaderStages, ShaderType,
            SpecializedMeshPipeline, SpecializedMeshPipelineError, SpecializedMeshPipelines,
            StorageTextureAccess, Texture, TextureAspect, TextureDescriptor, TextureDimension,
            TextureFormat, TextureSampleType, TextureUsages, TextureView, TextureViewDescriptor,
            TextureViewDimension, UniformBuffer,
        },
        renderer::{RenderContext, RenderDevice, RenderQueue},
        texture::{FallbackImage, Image},
        view::{ExtractedView, Msaa, RenderLayers, ViewDepthTexture, ViewTarget, VisibleEntities},
        Extract, ExtractSchedule, Render, RenderApp, RenderSet,
    },
    scene::SceneBundle,
    transform::components::{GlobalTransform, Transform},
    utils::{nonmax::NonMaxU32, FloatOrd, HashSet},
    window::Window, time::{Timer, TimerMode, Time},
};

use crate::{
    chunk::{ChunkHandle, ChunkMaterial},
    player::PlayerEntity,
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
    fn build(&self, app: &mut bevy::prelude::App) {
        fn setup(
            mut commands: Commands,
            mut images: ResMut<Assets<Image>>,
            mut materials: ResMut<Assets<TransientWorldMaterial>>,
            mut meshes: ResMut<Assets<Mesh>>,
        ) {
            let pos = Vec3::new(0.0, -100.0, 0.0);
            let voxel_size = 1.0 / 16.0;
            let cube_handle = meshes.add(Mesh::from(shape::Cube { size: 2.0 }));
            let material = materials.add(TransientWorldMaterial {
                side: 512,
                materials: images.add(ChunkHandle::material_image(vec![Vec4::ONE; 256])),
                chunk_position: pos,
                voxel_size,
            });

            commands.spawn((
                Name::new("transient world"),
                MaterialMeshBundle {
                    mesh: cube_handle,
                    material,
                    transform: Transform::from_translation(pos)
                        .with_scale(Vec3::NEG_ONE * voxel_size * 512.0 / 2.0),
                    ..Default::default()
                },
            ));
        }
        app.add_systems(Startup, setup);
    }

    fn finish(&self, app: &mut bevy::prelude::App) {
        let render_device = app.world.resource::<RenderDevice>();
        let render_queue = app.world.resource::<RenderQueue>();
        let world_data = WorldData::new_default(render_device, render_queue);

        let render_app = app.sub_app_mut(RenderApp);
        render_app.insert_resource(world_data);
        render_app.init_resource::<WorldDataUniforms>();
    }
}

#[derive(Resource)]
struct WorldData {
    render_pass_bind_group_layout: BindGroupLayout,
    render_pass_bind_group: BindGroup,

    depth_pass_bind_group_layout: BindGroupLayout,
    depth_pass_bind_group: BindGroup,

    voxelization_bind_group_layout: BindGroupLayout,
    voxelization_bind_group: BindGroup,

    // render the depth here to be used by other shaders
    // it is needed for fragment shader. only depth stencil can't have fragment shader
    depth_prepass_texture: Texture,
    depth_prepass_texture_view: TextureView,
    // depth stencil to allow more efficient rendering of chunk cubes (z buffer tests)
    depth_prepass_depth_stencil: Texture,
    depth_prepass_depth_stencil_view: TextureView,

    // TODO:
    transient_world_texture: Texture,
    transient_world_texture_view: TextureView,
    // NOTE: i am assuming we need to have this image just so that we can reuse the rendering pipeline for
    // voxelization. and we don't actually need this image's output. the real output will be done directly
    // the 3d world texture
    voxelizer_camera_target: Texture,
    voxelizer_camera_target_view: TextureView,
    voxelizer_camera_depth_stencil: Texture,
    voxelizer_camera_depth_stencil_view: TextureView,

    uniforms: UniformBuffer<WorldDataUniforms>,
    voxelization_timer: Timer,
}
impl WorldData {
    fn new_default(render_device: &RenderDevice, render_queue: &RenderQueue) -> Self {
        let render_pass_bind_group_layout = WorldData::render_pass_bind_group_layout(render_device);
        let depth_pass_bind_group_layout = WorldData::depth_pass_bind_group_layout(render_device);
        let voxelization_bind_group_layout =
            WorldData::voxelization_bind_group_layout(render_device);

        let depth_stencil_texture =
            Self::depth_pass_depth_stencil(render_device, Default::default());
        let depth_stencil_view =
            depth_stencil_texture.create_view(&TextureViewDescriptor::default());

        let depth_pass_texture = Self::depth_pass_texture(render_device, Default::default());
        let depth_pass_target_view =
            depth_pass_texture.create_view(&TextureViewDescriptor::default());

        let transient_world_size = Extent3d {
            width: 512,
            height: 512,
            depth_or_array_layers: 512,
        };
        let transient_world_texture =
            Self::transient_world_texture(render_device, transient_world_size);
        let transient_world_view =
            transient_world_texture.create_view(&TextureViewDescriptor::default());
        let voxelizer_camera_target = Self::voxelizer_camera_target(
            render_device,
            Extent3d {
                depth_or_array_layers: 1,
                ..transient_world_size
            },
        );
        let voxelizer_target_view =
            voxelizer_camera_target.create_view(&TextureViewDescriptor::default());
        let voxelizer_camera_depth_stencil_texture = Self::depth_pass_depth_stencil(
            render_device,
            Extent3d {
                depth_or_array_layers: 1,
                ..transient_world_size
            },
        );
        let voxelizer_camera_depth_stencil_view =
            voxelizer_camera_depth_stencil_texture.create_view(&TextureViewDescriptor::default());

        let mut uniforms = UniformBuffer::from(WorldDataUniforms::default());
        uniforms.write_buffer(render_device, render_queue);
        Self {
            render_pass_bind_group: render_device.create_bind_group(
                Some("chunk render pass bind group"),
                &render_pass_bind_group_layout,
                &[
                    BindGroupEntry {
                        binding: 0,
                        resource: uniforms.binding().unwrap(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::TextureView(&transient_world_view),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: BindingResource::TextureView(&depth_pass_target_view),
                    },
                ],
            ),
            render_pass_bind_group_layout,
            depth_pass_bind_group: render_device.create_bind_group(
                Some("depth prepass bind group"),
                &depth_pass_bind_group_layout,
                &[
                    BindGroupEntry {
                        binding: 0,
                        resource: uniforms.binding().unwrap(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::TextureView(&transient_world_view),
                    },
                ],
            ),
            depth_pass_bind_group_layout,
            voxelization_bind_group: render_device.create_bind_group(
                Some("voxelization bind group"),
                &voxelization_bind_group_layout,
                &[
                    BindGroupEntry {
                        binding: 0,
                        resource: uniforms.binding().unwrap(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::TextureView(&transient_world_view),
                    },
                ],
            ),
            voxelization_bind_group_layout,
            depth_prepass_texture: depth_pass_texture,
            depth_prepass_texture_view: depth_pass_target_view,
            depth_prepass_depth_stencil: depth_stencil_texture,
            depth_prepass_depth_stencil_view: depth_stencil_view,
            transient_world_texture,
            transient_world_texture_view: transient_world_view,
            voxelizer_camera_target,
            voxelizer_camera_target_view: voxelizer_target_view,
            voxelizer_camera_depth_stencil: voxelizer_camera_depth_stencil_texture,
            voxelizer_camera_depth_stencil_view,
            uniforms,
            voxelization_timer: Timer::from_seconds(1.0/10.0, TimerMode::Repeating),
        }
    }

    fn update(
        &mut self,
        render_device: &RenderDevice,
        render_queue: &RenderQueue,
        uniforms: &WorldDataUniforms,
        time: &Time,
    ) {
        self.voxelization_timer.tick(time.delta());
        if self.uniforms.get().screen_resolution != uniforms.screen_resolution {
            self.resize_depth_prepass(
                render_device,
                Extent3d {
                    width: uniforms.screen_resolution.x / uniforms.depth_prepass_scale_factor
                        + (uniforms.screen_resolution.x % uniforms.depth_prepass_scale_factor > 0)
                            as u32,
                    height: uniforms.screen_resolution.y / uniforms.depth_prepass_scale_factor
                        + (uniforms.screen_resolution.y % uniforms.depth_prepass_scale_factor > 0)
                            as u32,
                    depth_or_array_layers: 1,
                },
            );
        }

        self.uniforms.set(uniforms.clone());
        self.uniforms.write_buffer(render_device, render_queue);

        let bind_group = render_device.create_bind_group(
            Some("world data bind group"),
            &self.render_pass_bind_group_layout,
            &[
                BindGroupEntry {
                    binding: 0,
                    resource: self.uniforms.binding().unwrap(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&self.transient_world_texture_view),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(&self.depth_prepass_texture_view),
                },
            ],
        );
        self.render_pass_bind_group = bind_group;

        let bind_group = render_device.create_bind_group(
            Some("world data bind group"),
            &self.depth_pass_bind_group_layout,
            &[
                BindGroupEntry {
                    binding: 0,
                    resource: self.uniforms.binding().unwrap(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&self.transient_world_texture_view),
                },
            ],
        );
        self.depth_pass_bind_group = bind_group;
    }

    fn resize_depth_prepass(&mut self, render_device: &RenderDevice, size: Extent3d) {
        self.depth_prepass_texture = Self::depth_pass_texture(render_device, size);
        self.depth_prepass_texture_view = self
            .depth_prepass_texture
            .create_view(&TextureViewDescriptor::default());
        self.depth_prepass_depth_stencil = Self::depth_pass_depth_stencil(render_device, size);
        self.depth_prepass_depth_stencil_view = self
            .depth_prepass_depth_stencil
            .create_view(&TextureViewDescriptor::default());
    }

    fn transient_world_texture(render_device: &RenderDevice, size: Extent3d) -> Texture {
        render_device.create_texture(&TextureDescriptor {
            label: Some("transient world texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D3,
            format: TextureFormat::R32Uint,
            usage: TextureUsages::STORAGE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        })
    }
    fn voxelizer_camera_target(render_device: &RenderDevice, size: Extent3d) -> Texture {
        render_device.create_texture(&TextureDescriptor {
            label: Some("transient world camera target"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::R8Unorm,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[TextureFormat::R8Unorm],
        })
    }

    // TODO: can use AsBindGroup to generate this boilerplate
    fn render_pass_bind_group_layout(render_device: &RenderDevice) -> BindGroupLayout {
        render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("world data bind group"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT | ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: BufferSize::new(WorldDataUniforms::SHADER_SIZE.into()),
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::FRAGMENT | ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::ReadWrite,
                        format: TextureFormat::R32Uint,
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
        })
    }
    fn depth_pass_bind_group_layout(render_device: &RenderDevice) -> BindGroupLayout {
        render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("depth pass bind group"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT | ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: BufferSize::new(WorldDataUniforms::SHADER_SIZE.into()),
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::FRAGMENT | ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::ReadWrite,
                        format: TextureFormat::R32Uint,
                        view_dimension: TextureViewDimension::D3,
                    },
                    count: None,
                },
            ],
        })
    }
    fn voxelization_bind_group_layout(render_device: &RenderDevice) -> BindGroupLayout {
        render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("depth pass bind group"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT | ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: BufferSize::new(WorldDataUniforms::SHADER_SIZE.into()),
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::FRAGMENT | ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::ReadWrite,
                        format: TextureFormat::R32Uint,
                        view_dimension: TextureViewDimension::D3,
                    },
                    count: None,
                },
            ],
        })
    }
    fn depth_pass_texture(render_device: &RenderDevice, size: Extent3d) -> Texture {
        render_device.create_texture(&TextureDescriptor {
            label: Some("depth prepass texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::R32Float,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
    }
    fn depth_pass_depth_stencil(render_device: &RenderDevice, size: Extent3d) -> Texture {
        render_device.create_texture(&TextureDescriptor {
            label: Some("depth prepass depth stencil"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Depth32Float,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
    }
}

#[derive(Default, Debug, Clone, ShaderType, Resource)]
struct WorldDataUniforms {
    player_pos: Vec3,
    screen_resolution: UVec2,
    fov: f32,
    // v is voxel size
    // v / root2 cuz atleast 1 ray in a 2x2 ray gird must hit voxel
    // alpha is angle between 2 adjacent rays
    // v / root2 = t tan(alpha)
    //
    // theta is fov
    // d is distance of screen from camera
    // tan(theta/2) = 1/(2 * d)
    //
    // w is screen width in pixels
    // tan(alpha) = (1/w) * 1/d
    //
    // t = v * w * cot(theta/2) / (2 root2)
    unit_t_max: f32,
    depth_prepass_scale_factor: u32,
}
impl WorldDataUniforms {
    fn new(player_pos: Vec3, resolution: UVec2, fov: f32, depth_prepass_scale_factor: u32) -> Self {
        let theta = fov as f64;
        let t = (resolution.x as f64 / depth_prepass_scale_factor as f64)
            * (1.0 / (theta / 2.0).tan())
            * (1.0 / (2.0 * 2.0f64.sqrt()));
        Self {
            player_pos,
            screen_resolution: resolution,
            fov,
            unit_t_max: t as f32,
            depth_prepass_scale_factor,
        }
    }
}

// refer the source code of MaterialPlugin<M>
struct ChunkMaterialPlugin;
impl Plugin for ChunkMaterialPlugin {
    fn build(&self, app: &mut bevy::prelude::App) {
        app.init_asset::<ChunkMaterial>()
            .init_asset::<TransientWorldMaterial>()
            .add_plugins(ExtractInstancesPlugin::<AssetId<ChunkMaterial>>::extract_visible())
            .add_plugins(
                ExtractInstancesPlugin::<AssetId<TransientWorldMaterial>>::extract_visible(),
            );

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
            .init_resource::<DrawFunctions<TransientWorldPhaseItem>>()
            .add_render_command::<ChunkDepthPhaseItem, ChunkDepthPrepass<ChunkMaterial>>()
            .add_render_command::<ChunkRenderPhaseItem, DrawMaterial<ChunkMaterial>>()
            .add_render_command::<TransientWorldPhaseItem, TransientWorldRenderCommand>()
            .init_resource::<ExtractedMaterials<ChunkMaterial>>()
            .init_resource::<RenderMaterials<ChunkMaterial>>()
            .init_resource::<ExtractedMaterials<TransientWorldMaterial>>()
            .init_resource::<RenderMaterials<TransientWorldMaterial>>()
            .init_resource::<SpecializedMeshPipelines<ChunkPipeline>>()
            .init_resource::<SpecializedMeshPipelines<ChunkDepthPipeline>>()
            .init_resource::<SpecializedMeshPipelines<TransientWorldPipeline>>()
            .add_systems(ExtractSchedule, extract_materials::<ChunkMaterial>)
            .add_systems(ExtractSchedule, extract_materials::<TransientWorldMaterial>)
            .add_systems(
                Render,
                (
                    prepare_materials::<ChunkMaterial>
                        .in_set(RenderSet::PrepareAssets)
                        .after(prepare_assets::<Image>),
                    prepare_transient_materials
                        .in_set(RenderSet::PrepareAssets)
                        .after(prepare_assets::<Image>),
                    sort_phase_system::<ChunkDepthPhaseItem>.in_set(RenderSet::PhaseSort),
                    sort_phase_system::<ChunkRenderPhaseItem>.in_set(RenderSet::PhaseSort),
                    sort_phase_system::<TransientWorldPhaseItem>.in_set(RenderSet::PhaseSort),
                    queue_material_meshes::<ChunkMaterial>
                        .in_set(RenderSet::QueueMeshes)
                        .after(prepare_materials::<ChunkMaterial>),
                    queue_transient_material_meshes
                        .in_set(RenderSet::QueueMeshes)
                        .after(prepare_transient_materials),
                    (
                        batch_and_prepare_render_phase::<ChunkRenderPhaseItem, MeshPipeline>,
                        batch_and_prepare_render_phase::<ChunkDepthPhaseItem, MeshPipeline>,
                        batch_and_prepare_render_phase::<TransientWorldPhaseItem, MeshPipeline>,
                    )
                        .in_set(RenderSet::PrepareResources),
                ),
            );
    }
    fn finish(&self, app: &mut bevy::prelude::App) {
        let render_app = app.sub_app_mut(RenderApp);

        render_app
            .init_resource::<ChunkPipeline>()
            .init_resource::<TransientWorldPipeline>()
            .init_resource::<ChunkDepthPipeline>();

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
                .insert(RenderPhase::<TransientWorldPhaseItem>::default())
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
        &'static RenderPhase<TransientWorldPhaseItem>,
        &'static ViewTarget,
        &'static ViewDepthTexture,
        &'static Camera3d,
        &'static ExtractedCamera,
    );

    fn run(
        &self,
        graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        (depth_phase, render_phase, transient_phase, target, bevy_depth_view, cam3d, cam): QueryItem<
            Self::ViewQuery,
        >,
        world: &bevy::prelude::World,
    ) -> Result<(), NodeRunError> {
        let world_data = world.resource::<WorldData>();

        {
            let mut render_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
                label: Some("transient_world_render_pass"),
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

            transient_phase.render(&mut render_pass, world, graph.view_entity());
        }

        if render_phase.items.is_empty() {
            return Ok(());
        }

        {
            let mut render_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
                label: Some("chunk_depth_prepass"),
                // color_attachments: &[Some(target.get_color_attachment(Operations {
                //     load: LoadOp::Load,
                //     store: true,
                // }))],
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &world_data.depth_prepass_texture_view,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(Color::rgb_u8(0, 0, 0).into()),
                        store: true,
                    },
                })],
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
            let mut render_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
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

const VOXELIZATION_LAYER: u8 = 1;
pub struct VoxelizationPlugin;
impl Plugin for VoxelizationPlugin {
    fn build(&self, app: &mut bevy::prelude::App) {
        app.init_asset::<VoxelizationMaterial>()
            .add_plugins(
                ExtractInstancesPlugin::<AssetId<VoxelizationMaterial>>::extract_visible(),
            );

        fn setup(
            mut commands: Commands,
            mut meshes: ResMut<Assets<Mesh>>,
            mut materials: ResMut<Assets<VoxelizationMaterial>>,
            mut images: ResMut<Assets<Image>>,
        ) {
            for i in -3..0 {
                let mut transform = Transform::from_translation(Vec3::ZERO);
                match i {
                    -3 => transform.look_at(Vec3::X, Vec3::Y),
                    -2 => transform.look_at(Vec3::Z, Vec3::Y),
                    -1 => transform.look_at(Vec3::Y, Vec3::Z),
                    _ => unreachable!(),
                }
                commands.spawn((
                    Name::new("voxelization cam"),
                    VoxelizationCam,
                    Camera3dBundle {
                        camera: Camera {
                            order: i,
                            ..Default::default()
                        },
                        projection: Projection::Orthographic(OrthographicProjection {
                            near: -256.0,
                            far: 256.0,
                            scaling_mode: ScalingMode::Fixed {
                                width: 512.0,
                                height: 512.0,
                            },
                            ..Default::default()
                        }),
                        transform,
                        ..Default::default()
                    },
                    RenderLayers::layer(VOXELIZATION_LAYER),
                ));
            }

            commands.spawn((
                VoxelizationBundle {
                    material_mesh_bundle: MaterialMeshBundle {
                        mesh: meshes.add(Mesh::from(shape::Cube { size: 50.0 })),
                        material: materials.add(VoxelizationMaterial {
                            colors: images.add(Image::new_fill(
                                Extent3d {
                                    width: 256,
                                    height: 1,
                                    depth_or_array_layers: 1,
                                },
                                TextureDimension::D1,
                                &[255, 255, 255, 255],
                                TextureFormat::R8Uint,
                            )),
                        }),
                        transform: Transform::from_xyz(0.0, 0.0, 10.0),
                        ..Default::default()
                    },
                    ..Default::default()
                },
                crate::Rotates,
            ));
        }
        app.add_systems(Startup, setup);

        let Ok(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .add_render_graph_node::<ViewNodeRunner<VoxelizationRenderNode>>(
                CORE_3D,
                VoxelizationRenderNode::NAME,
            )
            .add_render_graph_edge(CORE_3D, MAIN_TRANSPARENT_PASS, VoxelizationRenderNode::NAME);
        // render_app.add_render_graph_node::<ClearNode>(CORE_3D, ClearNode::NAME)
        // .add_render_graph_edge(CORE_3D, ClearNode::NAME, CAMERA_DRIVER);
        let mut graph = render_app.world.resource_mut::<RenderGraph>();
        graph.add_node(ClearNode::NAME, ClearNode);
        graph.add_node_edge(ClearNode::NAME, CAMERA_DRIVER);

        #[allow(clippy::type_complexity)]
        fn extract_render_phase(
            mut commands: Commands,
            cameras_3d: Extract<Query<(Entity, &Camera), (With<Camera3d>, With<VoxelizationCam>)>>,
        ) {
            for (entity, camera) in &cameras_3d {
                if camera.is_active {
                    commands
                        .get_or_spawn(entity)
                        .insert(RenderPhase::<VoxelizationPhaseItem>::default());
                }
            }
        }

        render_app
            .add_systems(ExtractSchedule, extract_render_phase)
            .init_resource::<DrawFunctions<VoxelizationPhaseItem>>()
            .add_render_command::<VoxelizationPhaseItem, VoxelizationRenderCommand<VoxelizationMaterial>>()
            .init_resource::<ExtractedMaterials<VoxelizationMaterial>>()
            .init_resource::<RenderMaterials<VoxelizationMaterial>>()
            .init_resource::<SpecializedMeshPipelines<VoxelizationPipeline>>()
            .add_systems(ExtractSchedule, extract_materials::<VoxelizationMaterial>)
            .add_systems(
                Render,
        (
                    prepare_voxelization_materials::<VoxelizationMaterial>
                        .in_set(RenderSet::PrepareAssets)
                        .after(prepare_assets::<Image>),
                    sort_phase_system::<VoxelizationPhaseItem>.in_set(RenderSet::PhaseSort),
                    queue_voxelization_meshes::<VoxelizationMaterial>
                        .in_set(RenderSet::QueueMeshes)
                        .after(prepare_voxelization_materials::<VoxelizationMaterial>),
                    (
                        batch_and_prepare_render_phase::<VoxelizationPhaseItem, MeshPipeline>,
                    )
                        .in_set(RenderSet::PrepareResources),
                ));
    }
    fn finish(&self, app: &mut bevy::prelude::App) {
        let render_app = app.sub_app_mut(RenderApp);

        render_app
            .init_resource::<VoxelizationPipeline>()
            .init_resource::<ClearNodePipeline>();
    }
}
#[derive(Default)]
struct VoxelizationRenderNode;
impl VoxelizationRenderNode {
    const NAME: &'static str = "voxelization_render_node";
}
impl ViewNode for VoxelizationRenderNode {
    type ViewQuery = (
        &'static RenderPhase<VoxelizationPhaseItem>,
        &'static Camera3d,
        &'static ExtractedCamera,
        &'static ViewTarget,
        &'static ViewDepthTexture,
    );

    fn run(
        &self,
        graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        (phase, cam3d, cam, view, view_depth): QueryItem<Self::ViewQuery>,
        world: &bevy::prelude::World,
    ) -> Result<(), NodeRunError> {
        if phase.items.is_empty() {
            return Ok(());
        }

        let world_data = world.resource::<WorldData>();

        if !world_data.voxelization_timer.finished() {
            return Ok(());
        }

        {
            // TODO: clear this texture in a compute pass
            // TODO: execute this rendering + clearing across a bunch of frames
            // render_context.command_encoder().clear_texture(
            //     &world_data.transient_world_texture,
            //     &ImageSubresourceRange {
            //         aspect: TextureAspect::All,
            //         base_mip_level: 0,
            //         mip_level_count: None,
            //         base_array_layer: 0,
            //         array_layer_count: None,
            //     },
            // );
            let mut render_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
                label: Some("voxelization_render_pass"),
                color_attachments: &[
                    Some(RenderPassColorAttachment {
                        view: &world_data.voxelizer_camera_target_view,
                        resolve_target: None,
                        ops: Operations {
                            load: LoadOp::Clear(Color::rgb_u8(0, 0, 0).into()),
                            store: true,
                        },
                    }),
                    // Some(view.get_color_attachment(Operations {
                    //     load: LoadOp::Load,
                    //     store: true,
                    // })),
                ],
                // depth_stencil_attachment: None,
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                    view: &world_data.voxelizer_camera_depth_stencil_view,
                    // view: &view_depth.view,
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

            phase.render(&mut render_pass, world, graph.view_entity());
        }

        Ok(())
    }
}

#[derive(Resource)]
pub struct ClearNodePipeline {
    pipeline_id: CachedComputePipelineId,
}
impl FromWorld for ClearNodePipeline {
    fn from_world(world: &mut bevy::prelude::World) -> Self {
        let pipeline_cache = world.resource::<PipelineCache>();
        let world_data = world.resource::<WorldData>();
        let asset_server = world.resource::<AssetServer>();

        Self {
            pipeline_id: pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("clear node pipeline".into()),
                layout: vec![world_data.voxelization_bind_group_layout.clone()],
                push_constant_ranges: vec![],
                shader: asset_server.load("shaders/clear.wgsl"),
                shader_defs: vec![],
                entry_point: "clear_world".into(),
            }),
        }
    }
}

#[derive(Default)]
pub struct ClearNode;
impl ClearNode {
    const NAME: &'static str = "clear_node";
}
impl Node for ClearNode {
    fn update(&mut self, world: &mut bevy::prelude::World) {}
    fn run(
        &self,
        graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &bevy::prelude::World,
    ) -> Result<(), NodeRunError> {
        let pipeline = world.resource::<ClearNodePipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let world_data = world.resource::<WorldData>();

        if !world_data.voxelization_timer.finished() {
            return Ok(());
        }

        let Some(p) = pipeline_cache.get_compute_pipeline(pipeline.pipeline_id) else {
            return Ok(());
        };

        let mut pass = render_context
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor::default());

        pass.set_bind_group(0, &world_data.voxelization_bind_group, &[]);
        pass.set_pipeline(p);
        let dispatch_size = 512 / 4;
        pass.dispatch_workgroups(dispatch_size, dispatch_size, dispatch_size);

        Ok(())
    }
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn queue_voxelization_meshes<M: Material>(
    draw_functions: Res<DrawFunctions<VoxelizationPhaseItem>>,
    pipeline: Res<VoxelizationPipeline>,
    mut pipelines: ResMut<SpecializedMeshPipelines<VoxelizationPipeline>>,
    pipeline_cache: Res<PipelineCache>,
    render_meshes: Res<RenderAssets<Mesh>>,
    render_materials: Res<RenderMaterials<M>>,
    mut render_mesh_instances: ResMut<RenderMeshInstances>,
    render_material_instances: Res<RenderMaterialInstances<M>>,
    mut views: Query<(
        &ExtractedView,
        &VisibleEntities,
        Option<&Projection>,
        &mut RenderPhase<VoxelizationPhaseItem>,
    )>,
) where
    M::Data: PartialEq + Eq + std::hash::Hash + Clone,
{
    for (view, visible_entities, projection, mut phase) in &mut views {
        let draw = draw_functions.read().id::<VoxelizationRenderCommand<M>>();

        let mut view_key = MeshPipelineKey::from_hdr(view.hdr);

        if let Some(projection) = projection {
            view_key |= match projection {
                Projection::Perspective(_) => MeshPipelineKey::VIEW_PROJECTION_PERSPECTIVE,
                Projection::Orthographic(_) => MeshPipelineKey::VIEW_PROJECTION_ORTHOGRAPHIC,
            };
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

            let pipeline_id =
                pipelines.specialize(&pipeline_cache, &pipeline, mesh_key, &mesh.layout);
            let pipeline_id = match pipeline_id {
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

            phase.add(VoxelizationPhaseItem {
                entity: *visible_entity,
                draw_function: draw,
                pipeline: pipeline_id,
                distance,
                batch_range: 0..1,
                dynamic_offset: None,
            });
        }
    }
}
/// This system prepares all assets of the corresponding [`Material`] type
/// which where extracted this frame for the GPU.
#[allow(clippy::too_many_arguments)]
pub fn prepare_voxelization_materials<M: Material>(
    mut prepare_next_frame: Local<PrepareNextFrameMaterials<M>>,
    mut extracted_assets: ResMut<ExtractedMaterials<M>>,
    mut render_materials: ResMut<RenderMaterials<M>>,
    render_device: Res<RenderDevice>,
    images: Res<RenderAssets<Image>>,
    fallback_image: Res<FallbackImage>,
    pipeline: Res<VoxelizationPipeline>,
) {
    let queued_assets = std::mem::take(&mut prepare_next_frame.assets);
    for (id, material) in queued_assets.into_iter() {
        match prepare_voxelization_material(
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
        match prepare_voxelization_material(
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

fn prepare_voxelization_material<M: Material>(
    material: &M,
    render_device: &RenderDevice,
    images: &RenderAssets<Image>,
    fallback_image: &FallbackImage,
    pipeline: &VoxelizationPipeline,
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

#[derive(Component)]
pub struct VoxelizationCam;

#[derive(Component)]
pub struct Voxelizable;

#[derive(Bundle)]
pub struct VoxelizationBundle {
    v: Voxelizable,
    layer: RenderLayers,
    material_mesh_bundle: MaterialMeshBundle<VoxelizationMaterial>,
}
impl Default for VoxelizationBundle {
    fn default() -> Self {
        Self {
            v: Voxelizable,
            layer: RenderLayers::layer(VOXELIZATION_LAYER),
            material_mesh_bundle: Default::default(),
        }
    }
}

pub struct TransientWorldPhaseItem {
    pub distance: f32,
    pub pipeline: CachedRenderPipelineId,
    pub entity: Entity,
    pub draw_function: DrawFunctionId,
    pub batch_range: Range<u32>,
    pub dynamic_offset: Option<NonMaxU32>,
}
impl PhaseItem for TransientWorldPhaseItem {
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
impl CachedRenderPipelinePhaseItem for TransientWorldPhaseItem {
    #[inline]
    fn cached_pipeline(&self) -> CachedRenderPipelineId {
        self.pipeline
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

pub struct VoxelizationPhaseItem {
    pub distance: f32,
    pub pipeline: CachedRenderPipelineId,
    pub entity: Entity,
    pub draw_function: DrawFunctionId,
    pub batch_range: Range<u32>,
    pub dynamic_offset: Option<NonMaxU32>,
}
impl PhaseItem for VoxelizationPhaseItem {
    // does not matter how we sort here. as we want all triangles to be rendered anyway
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
impl CachedRenderPipelinePhaseItem for VoxelizationPhaseItem {
    #[inline]
    fn cached_pipeline(&self) -> CachedRenderPipelineId {
        self.pipeline
    }
}

fn extract_world_data(
    mut commands: Commands,
    player: Extract<Query<(&GlobalTransform, &Projection), With<PlayerEntity>>>,
    windows: Extract<Query<&Window>>,
) {
    let Ok(window) = windows.get_single() else {
        return;
    };
    let (player, Projection::Perspective(projection)) = player.single() else {
        panic!("player should have perspective projection. not orthographic.")
    };
    let width = window.resolution.physical_width() as _;
    let height = window.resolution.physical_height() as _;

    commands.insert_resource(WorldDataUniforms::new(
        player.translation(),
        UVec2::new(width, height),
        projection.fov,
        4,
    ));
}

fn update_world_data(
    mut world_data: ResMut<WorldData>,
    world_data_uniforms: Res<WorldDataUniforms>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    time: Res<Time>,
) {
    world_data.update(&render_device, &render_queue, &world_data_uniforms, &time);
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

            let render_pipeline_id =
                pipelines.specialize(&pipeline_cache, &material_pipeline, mesh_key, &mesh.layout);
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
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn queue_transient_material_meshes(
    draw_functions: Res<DrawFunctions<TransientWorldPhaseItem>>,
    material_pipeline: Res<ChunkPipeline>,
    pipeline: Res<TransientWorldPipeline>,
    mut pipelines: ResMut<SpecializedMeshPipelines<TransientWorldPipeline>>,
    pipeline_cache: Res<PipelineCache>,
    render_meshes: Res<RenderAssets<Mesh>>,
    render_materials: Res<RenderMaterials<TransientWorldMaterial>>,
    mut render_mesh_instances: ResMut<RenderMeshInstances>,
    render_material_instances: Res<RenderMaterialInstances<TransientWorldMaterial>>,
    mut views: Query<(
        &ExtractedView,
        &VisibleEntities,
        Option<&Camera3d>,
        Option<&TemporalJitter>,
        Option<&Projection>,
        &mut RenderPhase<TransientWorldPhaseItem>,
    )>,
) {
    for (view, visible_entities, camera_3d, temporal_jitter, projection, mut phase) in &mut views {
        let draw = draw_functions.read().id::<TransientWorldRenderCommand>();

        let mut view_key = MeshPipelineKey::from_hdr(view.hdr);

        if temporal_jitter.is_some() {
            view_key |= MeshPipelineKey::TEMPORAL_JITTER;
        }

        if let Some(projection) = projection {
            view_key |= match projection {
                Projection::Perspective(_) => MeshPipelineKey::VIEW_PROJECTION_PERSPECTIVE,
                Projection::Orthographic(_) => MeshPipelineKey::VIEW_PROJECTION_ORTHOGRAPHIC,
            };
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

            let pipeline_id =
                pipelines.specialize(&pipeline_cache, &pipeline, mesh_key, &mesh.layout);
            let pipeline_id = match pipeline_id {
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

            phase.add(TransientWorldPhaseItem {
                entity: *visible_entity,
                draw_function: draw,
                pipeline: pipeline_id,
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
#[allow(clippy::too_many_arguments)]
pub fn prepare_transient_materials(
    mut prepare_next_frame: Local<PrepareNextFrameMaterials<TransientWorldMaterial>>,
    mut extracted_assets: ResMut<ExtractedMaterials<TransientWorldMaterial>>,
    mut render_materials: ResMut<RenderMaterials<TransientWorldMaterial>>,
    render_device: Res<RenderDevice>,
    images: Res<RenderAssets<Image>>,
    fallback_image: Res<FallbackImage>,
    pipeline: Res<TransientWorldPipeline>,
) {
    let queued_assets = std::mem::take(&mut prepare_next_frame.assets);
    for (id, material) in queued_assets.into_iter() {
        match prepare_transient_material(
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
        match prepare_transient_material(
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

fn prepare_transient_material(
    material: &TransientWorldMaterial,
    render_device: &RenderDevice,
    images: &RenderAssets<Image>,
    fallback_image: &FallbackImage,
    pipeline: &TransientWorldPipeline,
) -> Result<PreparedMaterial<TransientWorldMaterial>, AsBindGroupError> {
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
        desc.layout.insert(3, self.world_bind_group_layout.clone());
        let frag = desc.fragment.as_mut().unwrap();
        frag.shader = self.chunk_fragment_shader.clone();
        frag.shader_defs
            .push(ShaderDefVal::Bool("CHUNK_DEPTH_PREPASS".into(), true));
        frag.targets[0] = Some(ColorTargetState {
            format: TextureFormat::R32Float,
            blend: None,
            write_mask: ColorWrites::RED,
        });
        Ok(desc)
    }
}

#[derive(Resource)]
pub struct VoxelizationPipeline {
    mesh_pipeline: MeshPipeline,
    material_bind_group_layout: BindGroupLayout,
    world_bind_group_layout: BindGroupLayout,
    chunk_fragment_shader: Handle<Shader>,
}
impl FromWorld for VoxelizationPipeline {
    fn from_world(world: &mut bevy::prelude::World) -> Self {
        let world_data = world.resource::<WorldData>();
        let asset_server = world.resource::<AssetServer>();
        let render_device = world.resource::<RenderDevice>();
        let chunk_fragment_shader = asset_server.load("shaders/voxelization.wgsl");
        Self {
            mesh_pipeline: world.resource::<MeshPipeline>().clone(),
            world_bind_group_layout: world_data.voxelization_bind_group_layout.clone(),
            material_bind_group_layout: VoxelizationMaterial::bind_group_layout(render_device),
            chunk_fragment_shader,
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
        desc.primitive.cull_mode = None;
        desc.layout
            .insert(1, self.material_bind_group_layout.clone());
        desc.layout.insert(3, self.world_bind_group_layout.clone());
        desc.vertex.shader = self.chunk_fragment_shader.clone();
        let frag = desc.fragment.as_mut().unwrap();
        frag.shader = self.chunk_fragment_shader.clone();
        frag.shader_defs
            .push(ShaderDefVal::Bool("VOXELIZATION".into(), true));
        frag.targets[0] = Some(ColorTargetState {
            format: TextureFormat::R8Unorm,
            blend: None,
            write_mask: ColorWrites::RED,
        });
        Ok(desc)
    }
}

#[derive(Resource)]
pub struct TransientWorldPipeline {
    mesh_pipeline: MeshPipeline,
    world_bind_group_layout: BindGroupLayout,
    material_bind_group_layout: BindGroupLayout,
    fragment_shader: Handle<Shader>,
}
impl FromWorld for TransientWorldPipeline {
    fn from_world(world: &mut bevy::prelude::World) -> Self {
        let world_data = world.resource::<WorldData>();
        let asset_server = world.resource::<AssetServer>();
        let render_device = world.resource::<RenderDevice>();
        let fragment_shader = asset_server.load("shaders/transient_world.wgsl");
        Self {
            mesh_pipeline: world.resource::<MeshPipeline>().clone(),
            world_bind_group_layout: world_data.voxelization_bind_group_layout.clone(),
            material_bind_group_layout: TransientWorldMaterial::bind_group_layout(render_device),
            fragment_shader,
        }
    }
}
impl SpecializedMeshPipeline for TransientWorldPipeline {
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
        desc.layout.insert(3, self.world_bind_group_layout.clone());
        let frag = desc.fragment.as_mut().unwrap();
        frag.shader = self.fragment_shader.clone();
        frag.shader_defs
            .push(ShaderDefVal::Bool("TRANSIENT_WORLD".into(), true));
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

#[derive(Asset, TypePath, AsBindGroup, Debug, Clone, Component, ExtractComponent)]
struct VoxelizationMaterial {
    #[texture(0, sample_type = "u_int", dimension = "1d")]
    // TODO: R8Uint color pallet indices
    colors: Handle<Image>,
}
impl Material for VoxelizationMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/voxelize.wgsl".into()
    }

    fn alpha_mode(&self) -> AlphaMode {
        // AlphaMode::Blend
        AlphaMode::Add
    }
}
type VoxelizationRenderCommand<M> = (
    SetItemPipeline,
    SetMeshViewBindGroup<0>,
    SetMaterialBindGroup<M, 1>,
    SetMeshBindGroup<2>,
    SetWorldVoxelizationBindGroup<3>,
    DrawMesh,
);

#[derive(Asset, TypePath, AsBindGroup, Debug, Clone, Component, ExtractComponent)]
pub struct TransientWorldMaterial {
    #[uniform(0)]
    pub side: u32,
    #[texture(2, dimension = "1d", sample_type = "float", filterable = false)]
    pub materials: Handle<Image>,
    #[uniform(5)]
    pub chunk_position: Vec3,
    #[uniform(6)]
    pub voxel_size: f32,
    // #[texture(1, sample_type = "u_int", dimension = "3d")]
    // pub voxels: Handle<Image>,
    // #[texture(7, sample_type = "u_int", dimension = "3d")]
    // pub voxels_mip1: Handle<Image>,
    // #[texture(8, sample_type = "u_int", dimension = "3d")]
    // pub voxels_mip2: Handle<Image>,
}

impl Material for TransientWorldMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/transient_world.wgsl".into()
    }

    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Opaque
    }
}
type TransientWorldRenderCommand = (
    SetItemPipeline,
    SetMeshViewBindGroup<0>,
    SetMaterialBindGroup<TransientWorldMaterial, 1>,
    SetMeshBindGroup<2>,
    SetWorldVoxelizationBindGroup<3>,
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
struct SetWorldVoxelizationBindGroup<const I: usize>;
impl<P: PhaseItem, const I: usize> RenderCommand<P> for SetWorldVoxelizationBindGroup<I> {
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

        pass.set_bind_group(I, &world.voxelization_bind_group, &[]);

        RenderCommandResult::Success
    }
}
