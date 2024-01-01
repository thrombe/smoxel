#import bevy_pbr::forward_io::VertexOutput

struct CustomMaterial {
    color: vec4<f32>,
};

@group(1) @binding(0) var<uniform> side: u32;
@group(1) @binding(1) var voxels: texture_3d<u32>; // 4 voxels per u32 TODO: little endian or big endian ? :/
@group(1) @binding(2) var materials: texture_1d<f32>; // idk how these textures work :(
@group(1) @binding(3) var<uniform> pos: vec3<f32>;
@group(1) @binding(4) var<uniform> resolution: vec2<f32>;
@group(1) @binding(5) var<uniform> chunk_pos: vec3<f32>;
@group(1) @binding(6) var<uniform> chunk_size: f32;

@fragment
fn fragment(
    mesh: VertexOutput,
) -> @location(0) vec4<f32> {
    // let uv = mesh.position.xy/1000.0;
    // return vec4<f32>(((uv.x * uv.x + uv.y * uv.y)*10.0)%1.0, uv.x, uv.y, 1.0);
    // return vec4<f32>(mesh.world_position.xyz/128.0, 1.0);
    // return vec4<f32>(pos.xyz/128.0, 1.0);

    let screen_uv = mesh.position.xy/resolution.xy;
    let ray_pos = mesh.world_position.xyz;
    let ray_origin = pos.xyz;
    let ray_dir = normalize(ray_pos - ray_origin);

    // return vec4<f32>(mesh.position.xy/resolution.xy, 0.0, 1.0);
    var color = chunk_pos;
    color = ray_pos - chunk_pos;
    // color = abs(color);
    // color -= chunk_size;
    color = color / chunk_size;
    // color /= 2.0;
    // color = color % 1.1;
    // color -= chunk_size*0.9;
    // color = normalize(color);
    var alpha = 1.0;
    // alpha = length(color);
    // alpha -= 1.0;
    // alpha = 0.7;
    return vec4<f32>(color, alpha);
}
