
struct WorldDataUniforms {
    player_pos: vec3<f32>,
    screen_resolution: vec2<u32>,
    fov: f32,
    unit_t_max: f32,
    depth_prepass_scale_factor: u32,
}

@group(0) @binding(0) var<uniform> world_data: WorldDataUniforms;
// - [WebGPU Shading Language](https://www.w3.org/TR/WGSL/#storage-texel-formats)
@group(0) @binding(1) var transient_world_texture: texture_storage_3d<r32uint, read_write>;
@group(0) @binding(2) var depth_texture: texture_2d<f32>;

@compute @workgroup_size(4, 4, 4)
fn clear_world(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    textureStore(transient_world_texture, invocation_id, vec4(0u));
}


