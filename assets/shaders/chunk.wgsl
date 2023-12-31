#import bevy_pbr::forward_io::VertexOutput
// #import bevy_pbr::prepass_io::VertexOutput

struct CustomMaterial {
    color: vec4<f32>,
};

// @group(1) @binding(0) var<uniform> material: CustomMaterial;
// @group(1) @binding(1) var base_color_texture: texture_2d<f32>;
// @group(1) @binding(2) var base_color_sampler: sampler;

@group(1) @binding(0) var<uniform> side: u32;
@group(1) @binding(1) var voxels: texture_3d<u32>; // 4 voxels per u32
@group(1) @binding(2) var materials: texture_1d<f32>; // idk how these textures work :(

@fragment
fn fragment(
    mesh: VertexOutput,
    // in: prepass_io::VertexOutput,
) -> @location(0) vec4<f32> {
    // return vec4<f32>(0.8, 0.8, 0.8, 1.0) * textureSample(base_color_texture, base_color_sampler, mesh.uv) * COLOR_MULTIPLIER;
    // if (mesh.position.x > 500.0) {
        // return vec4<f32>(mesh.position.x/1000.0, mesh.position.y/1000.0, 0.0, 1.0);
    // }
    // let uv = mesh.position.xy/1000.0;
    // return vec4<f32>(((uv.x * uv.x + uv.y * uv.y)*10.0)%1.0, uv.x, uv.y, 1.0);
    return vec4<f32>(mesh.world_position.xyz/128.0, 1.0);
    // return vec4<f32>(mesh.position.xyz/128.0, 1.0);
}
