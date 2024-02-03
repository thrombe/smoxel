#import bevy_pbr::{mesh_functions, forward_io::VertexOutput}

struct WorldDataUniforms {
    player_pos: vec3<f32>,
    screen_resolution: vec2<u32>,
    fov: f32,
    unit_t_max: f32,
    depth_prepass_scale_factor: u32,
}

@group(3) @binding(0) var<uniform> world_data: WorldDataUniforms;
// - [WebGPU Shading Language](https://www.w3.org/TR/WGSL/#storage-texel-formats)
@group(3) @binding(1) var transient_world_texture: texture_storage_3d<r32uint, read_write>;
@group(3) @binding(2) var depth_texture: texture_2d<f32>;

struct Output {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
}

struct Vertex {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
    @location(2) uv: vec2<f32>,
}

@vertex
fn vertex(vertex: Vertex) -> VertexOutput {
    var out: VertexOutput;
    var model = mesh_functions::get_model_matrix(vertex.instance_index);
    out.world_position = mesh_functions::mesh_position_local_to_world(model, vec4<f32>(vertex.position, 1.0));
    out.position = mesh_functions::mesh_position_local_to_clip(model, vec4<f32>(vertex.position, 1.0));
    out.uv = vertex.uv;
    return out;
}

@fragment
fn fragment(mesh: VertexOutput) -> Output {
    var out: Output;

    textureStore(transient_world_texture, vec3<i32>(mesh.world_position.xyz) + 128, vec4(1u));

    // out.color = vec4(mesh.position.xyz / 2000.0, 0.3);
    out.color = vec4(abs(mesh.world_position.xyz) / 100.0, 1.0);
    out.color.w = 0.0;
    out.depth = 0.0;
    // out.color.x = 10.0/(abs(mesh.world_position.z));
    return out;
}
