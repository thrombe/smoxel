//! Bevy has an optional prepass that is controlled per-material. A prepass is a rendering pass that runs before the main pass.
//! It will optionally generate various view textures. Currently it supports depth, normal, and motion vector textures.
//! The textures are not generated for any material using alpha blending.

use bevy::{
    core_pipeline::prepass::{DepthPrepass, MotionVectorPrepass, NormalPrepass},
    input::mouse::MouseMotion,
    pbr::{wireframe::WireframePlugin, NotShadowCaster, PbrPlugin},
    prelude::*,
    reflect::TypePath,
    render::render_resource::{AsBindGroup, ShaderRef, ShaderType},
    window::{CursorGrabMode, PrimaryWindow},
};

fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins.set(PbrPlugin {
                // The prepass is enabled by default on the StandardMaterial,
                // but you can disable it if you need to.
                //
                // prepass_enabled: false,
                ..default()
            }),
            MaterialPlugin::<CustomMaterial>::default(),
            MaterialPlugin::<PrepassOutputMaterial> {
                // This material only needs to read the prepass textures,
                // but the meshes using it should not contribute to the prepass render, so we can disable it.
                prepass_enabled: false,
                ..default()
            },
        ))
        .add_state::<AppState>()
        .add_state::<ControlsState>()
        .add_plugins(WireframePlugin)
        .add_plugins(player::Player)
        .add_plugins(spectator::Spectator)
        .add_plugins(voxel::VoxelPlugin)
        .add_systems(Startup, setup)
        .add_systems(Update, (rotate, toggle_prepass_view))
        // Disabling MSAA for maximum compatibility. Shader prepass with MSAA needs GPU capability MULTISAMPLED_SHADING
        .insert_resource(Msaa::Off)
        .run();
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

mod voxel {
    use bevy::render::mesh::{Indices, VertexAttributeValues};
    use bevy::render::render_resource::PrimitiveTopology;
    use block_mesh::ndshape::ConstShape3u32;
    use block_mesh::{greedy_quads, GreedyQuadsBuffer, MergeVoxel, Voxel, VoxelVisibility};
    use noise::NoiseFn;

    use super::*;

    #[derive(Clone, Copy, Eq, PartialEq)]
    struct BoolVoxel(bool);
    impl Voxel for BoolVoxel {
        fn get_visibility(&self) -> VoxelVisibility {
            if !self.0 {
                VoxelVisibility::Empty
            } else {
                VoxelVisibility::Opaque
            }
        }
    }

    impl MergeVoxel for BoolVoxel {
        type MergeValue = Self;

        fn merge_value(&self) -> Self::MergeValue {
            *self
        }
    }

    #[derive(Resource)]
    pub struct VoxelWorld {
        noise: noise::Perlin,
    }

    pub struct VoxelPlugin;
    impl Plugin for VoxelPlugin {
        fn build(&self, app: &mut App) {
            app.add_systems(Startup, setup_voxel_plugin)
                .add_systems(OnEnter(AppState::Playing), test_spawn);
        }
    }

    fn setup_voxel_plugin(mut commands: Commands, mut game_state: ResMut<NextState<AppState>>) {
        let perlin = noise::Perlin::new(3243);
        let world = VoxelWorld { noise: perlin };
        commands.insert_resource(world);
        game_state.set(AppState::Playing);
    }

    fn test_spawn(
        mut commands: Commands,
        world: ResMut<VoxelWorld>,
        mut meshes: ResMut<Assets<Mesh>>,
        mut materials: ResMut<Assets<StandardMaterial>>,
    ) {
        const SIDEU32: u32 = 32;
        let side = SIDEU32 as usize;
        let mut voxels = vec![BoolVoxel(false); side * side * side];
        for z in 1..side - 1 {
            for y in 1..side - 1 {
                for x in 1..side - 1 {
                    let scale = 0.09;
                    let xf = x as f64 * scale;
                    let yf = y as f64 * scale;
                    let zf = z as f64 * scale;
                    let v = world.noise.get([xf, yf, zf]);
                    // if ((x*x + y*y + z*z) as f64) < 10.0_f64.powf(3.0) {
                    if v > 0.2 {
                        voxels[z * side * side + y * side + x] = BoolVoxel(true);
                    }
                }
            }
        }
        let mut buffer = GreedyQuadsBuffer::new(voxels.len());
        type ChunkShape = ConstShape3u32<SIDEU32, SIDEU32, SIDEU32>;
        let faces = &block_mesh::RIGHT_HANDED_Y_UP_CONFIG.faces;
        greedy_quads(
            &voxels,
            &ChunkShape {},
            [0; 3],
            [SIDEU32 - 1; 3],
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
                positions.extend_from_slice(&face.quad_mesh_positions(&quad, 1.0));
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

        let mesh_handle = meshes.add(render_mesh);

        commands.spawn((MaterialMeshBundle {
            mesh: mesh_handle.clone(),
            material: materials.add(StandardMaterial {
                base_color: Color::rgb(0.8, 0.8, 0.8),
                alpha_mode: AlphaMode::Opaque,
                ..Default::default()
            }),
            transform: Transform::from_xyz(0.0, 0.0, 0.0),
            ..Default::default()
        },));
    }
}

mod player {
    use super::*;

    #[derive(Component)]
    pub struct PlayerEntity;

    pub struct Player;
    impl Plugin for Player {
        fn build(&self, app: &mut App) {
            app.add_systems(Startup, setup).add_systems(
                Update,
                update.run_if(in_state(ControlsState::Player)),
            );
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
        }

        let (mut transform, _) = player.single_mut();
        if wind.cursor.grab_mode != CursorGrabMode::Locked {
            if mouse_keys.just_pressed(MouseButton::Left) {
                wind.cursor.visible = false;
                wind.cursor.grab_mode = CursorGrabMode::Confined;
            } else if mouse_keys.just_released(MouseButton::Left) {
                wind.cursor.visible = true;
                wind.cursor.grab_mode = CursorGrabMode::None;
            } else if mouse_keys.pressed(MouseButton::Left) {
                let pos = Vec2::new(
                    wind.resolution.width() / 2.0,
                    wind.resolution.height() / 2.0,
                );
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

        // TODO: clamp y rotation to +- Vec3::Y
        let quat = Quat::from_axis_angle(Vec3::Y, -mouse_delta.x)
            * Quat::from_axis_angle(transform.local_x(), -mouse_delta.y);
        transform.rotate(quat);

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
        keys: Res<Input<KeyCode>>,
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
        }

        let (mut transform, _) = spectator.single_mut();
        if wind.cursor.grab_mode != CursorGrabMode::Locked {
            if mouse_keys.just_pressed(MouseButton::Left) {
                wind.cursor.visible = false;
                wind.cursor.grab_mode = CursorGrabMode::Confined;
            } else if mouse_keys.just_released(MouseButton::Left) {
                wind.cursor.visible = true;
                wind.cursor.grab_mode = CursorGrabMode::None;
            } else if mouse_keys.pressed(MouseButton::Left) {
                let pos = Vec2::new(
                    wind.resolution.width() / 2.0,
                    wind.resolution.height() / 2.0,
                );
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

        // TODO: clamp y rotation to +- Vec3::Y
        let quat = Quat::from_axis_angle(Vec3::Y, -mouse_delta.x)
            * Quat::from_axis_angle(transform.local_x(), -mouse_delta.y);
        transform.rotate(quat);

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
    mut depth_materials: ResMut<Assets<PrepassOutputMaterial>>,
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
    commands.spawn((
        MaterialMeshBundle {
            mesh: meshes.add(shape::Quad::new(Vec2::new(20.0, 20.0)).into()),
            material: depth_materials.add(PrepassOutputMaterial {
                settings: ShowPrepassSettings::default(),
            }),
            transform: Transform::from_xyz(-0.75, 1.25, 3.0)
                .looking_at(Vec3::new(2.0, -2.5, -5.0), Vec3::Y),
            ..default()
        },
        NotShadowCaster,
    ));

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

    commands.spawn(
        TextBundle::from_sections(vec![
            TextSection::new("Prepass Output: transparent\n", style.clone()),
            TextSection::new("\n\n", style.clone()),
            TextSection::new("Controls\n", style.clone()),
            TextSection::new("---------------\n", style.clone()),
            TextSection::new("Space - Change output\n", style),
        ])
        .with_style(Style {
            position_type: PositionType::Absolute,
            top: Val::Px(10.0),
            left: Val::Px(10.0),
            ..default()
        }),
    );
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

#[derive(Debug, Clone, Default, ShaderType)]
struct ShowPrepassSettings {
    show_depth: u32,
    show_normals: u32,
    show_motion_vectors: u32,
    padding_1: u32,
    padding_2: u32,
}

// This shader simply loads the prepass texture and outputs it directly
#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub struct PrepassOutputMaterial {
    #[uniform(0)]
    settings: ShowPrepassSettings,
}

impl Material for PrepassOutputMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/show_prepass.wgsl".into()
    }

    // This needs to be transparent in order to show the scene behind the mesh
    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Blend
    }
}

/// Every time you press space, it will cycle between transparent, depth and normals view
fn toggle_prepass_view(
    mut prepass_view: Local<u32>,
    keycode: Res<Input<KeyCode>>,
    material_handle: Query<&Handle<PrepassOutputMaterial>>,
    mut materials: ResMut<Assets<PrepassOutputMaterial>>,
    mut text: Query<&mut Text>,
) {
    if keycode.just_pressed(KeyCode::Space) {
        *prepass_view = (*prepass_view + 1) % 4;

        let label = match *prepass_view {
            0 => "transparent",
            1 => "depth",
            2 => "normals",
            3 => "motion vectors",
            _ => unreachable!(),
        };
        let mut text = text.single_mut();
        text.sections[0].value = format!("Prepass Output: {label}\n");
        for section in &mut text.sections {
            section.style.color = Color::WHITE;
        }

        let handle = material_handle.single();
        let mat = materials.get_mut(handle).unwrap();
        mat.settings.show_depth = (*prepass_view == 1) as u32;
        mat.settings.show_normals = (*prepass_view == 2) as u32;
        mat.settings.show_motion_vectors = (*prepass_view == 3) as u32;
    }
}
