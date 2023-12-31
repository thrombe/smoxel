#import bevy_pbr::forward_io::VertexOutput
// we can import items from shader modules in the assets folder with a quoted path
#import "shaders/custom_material_import.wgsl"::COLOR_MULTIPLIER

struct CustomMaterial {
    color: vec4<f32>,
};

@group(1) @binding(0) var<uniform> material: CustomMaterial;
@group(1) @binding(1) var base_color_texture: texture_2d<f32>;
@group(1) @binding(2) var base_color_sampler: sampler;

@fragment
fn fragment(
    mesh: VertexOutput,
) -> @location(0) vec4<f32> {
    // return material.color * textureSample(base_color_texture, base_color_sampler, mesh.uv) * COLOR_MULTIPLIER;
    // if (mesh.position.x > 500.0) {
        // return vec4<f32>(mesh.position.x/1000.0, mesh.position.y/1000.0, 0.0, 1.0);
    // }
    let uv = mesh.position.xy/1000.0;
    return vec4<f32>(((uv.x * uv.x + uv.y * uv.y)*10.0)%1.0, uv.x, uv.y, 1.0);
}
