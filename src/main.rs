use bevy::{
    diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin},
    pbr::{wireframe::WireframePlugin, PbrPlugin},
    prelude::*,
    reflect::TypePath,
    render::{
        render_resource::{AsBindGroup, ShaderRef},
        RenderPlugin,
    },
};

#[cfg(feature = "dev")]
use bevy_inspector_egui::{
    bevy_egui::{self, EguiContexts},
    quick::WorldInspectorPlugin,
};

mod chunk;
mod player;
mod render;
mod vox;

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
    .add_plugins(player::spectator::Spectator)
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
    // plane
    commands.spawn(PbrBundle {
        mesh: meshes.add(shape::Plane::from_size(5.0).into()),
        material: std_materials.add(Color::rgb(0.3, 0.5, 0.3).into()),
        ..default()
    });

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
}

#[derive(Component)]
struct Rotates;

fn rotate(mut q: Query<&mut Transform, With<Rotates>>, time: Res<Time>) {
    for mut t in q.iter_mut() {
        let rot = (time.elapsed_seconds().sin() * 0.5 + 0.5) * std::f32::consts::PI * 2.0;
        t.rotation = Quat::from_rotation_z(rot);
    }
}
