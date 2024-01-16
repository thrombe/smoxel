#import bevy_pbr::forward_io::VertexOutput

@group(1) @binding(0) var<uniform> side: u32;
@group(1) @binding(5) var<uniform> chunk_pos: vec3<f32>;
@group(1) @binding(6) var<uniform> chunk_size: f32;

@group(1) @binding(1) var voxels: texture_3d<u32>;
@group(1) @binding(7) var voxels_mip1: texture_3d<u32>;
@group(1) @binding(8) var voxels_mip2: texture_3d<u32>;
@group(1) @binding(2) var materials: texture_1d<f32>;

struct WorldDataUniforms {
    player_pos: vec3<f32>,
    screen_resolution: vec2<u32>,
}

@group(3) @binding(0) var<uniform> world_data: WorldDataUniforms;
@group(3) @binding(2) var depth_texture: texture_2d<f32>;

// returns voxel material in the rightmost byte
fn get_voxel(pos: vec3<f32>) -> u32 {
    if (any(pos >= f32(side) || pos < 0.0)) {
        return 0u;
    }
    
    var coords = vec3<u32>(pos);
    let xyzw = coords.x % 4u;
    coords.x /= 4u;

    // loading textures automatically break the u32 into a vec4<u32> 1 byte each channel (~~ vec4<u8>)
    var voxel = textureLoad(voxels, coords, 0);

    return voxel[xyzw];
}

fn get_color(index: u32) -> vec4<f32> {
    return textureLoad(materials, index, 0);
}


fn get_mip1(pos: vec3<f32>) -> vec2<u32> {
    var coords = vec3<u32>(pos);
    var voxel = textureLoad(voxels_mip1, coords/4u, 0);
    // return vec4<u32>(voxel.x & 31u, voxel.x >> 16u, voxel.y & 31u, voxel.y >> 16u);
    return voxel.xy;
}
fn get_mip2(pos: vec3<f32>) -> vec2<u32> {
    var coords = vec3<u32>(pos);
    var voxel = textureLoad(voxels_mip2, coords/(4u * 8u), 0);
    return voxel.xy;
}

fn unpackBytes(t: u32) -> vec4<u32> {
    return (t >> vec4<u32>(0u, 8u, 16u, 24u)) & 15u;
}

struct Mip {
    z0: vec4<u32>,
    z1: vec4<u32>,
}

fn getMipByte(mip: vec2<u32>, index: u32) -> u32 {
    var component = 0u;
    switch (index) {
        case 0u {
            component = mip.x & 255u;
        }
        case 1u {
            component = mip.x & (255u << 8u);
        }
        case 2u {
            component = mip.x & (255u << 16u);
        }
        case 3u {
            component = mip.x & (255u << 24u);
        }
        case 4u {
            component = mip.y & 255u;
        }
        case 5u {
            component = mip.y & (255u << 8u);
        }
        case 6u {
            component = mip.y & (255u << 16u);
        }
        case 7u {
            component = mip.y & (255u << 24u);
        }
        default: {
            // discard;
        }
    }
    return component;
}
fn getMipBit(mip: u32, index: u32) -> u32 {
    var mask = 1u << index;
    mask = mask | mask << 8u | mask << 16u | mask << 24u;
    return mip & mask;
}

fn get_voxel_unchecked(pos: vec3<i32>) -> u32 {
    var coords = pos;
    let xyzw = coords.x % 4;
    coords.x /= 4;

    // loading textures automatically break the u32 into a vec4<u32> 1 byte each channel (~~ vec4<u8>)
    var voxel = textureLoad(voxels, coords, 0);

    return voxel[xyzw];
}

fn get_mip1_unchecked(pos: vec3<i32>) -> vec2<u32> {
    var voxel = textureLoad(voxels_mip1, pos/4, 0);
    return voxel.xy;
}

fn get_mip2_unckecked(pos: vec3<i32>) -> vec2<u32> {
    var voxel = textureLoad(voxels_mip2, pos/(4 * 8), 0);
    return voxel.xy;
}

fn mip0(march: vec3<f32>) -> bool {
    var mip = get_mip1(march);
    var marchu = vec3<i32>(march);
    var mod4 = marchu % 4;
    var m = vec3<u32>(mod4 > 1) << vec3(0u, 1u, 2u);
    var index = m.x | m.y | m.z;
    let comp = getMipByte(mip, index);
    var mod2 = marchu % 2;
    m = vec3<u32>(mod2 > 0) << vec3(0u, 1u, 2u);
    index = m.x | m.y | m.z;
    let voxel = getMipBit(comp, index);
    return voxel != 0u;
}
fn mip1(march: vec3<f32>) -> bool {
    var mip = get_mip1(march);
    var marchu = vec3<i32>(march);
    var mod4 = marchu % 4;
    var m = vec3<u32>(mod4 > 1) << vec3(0u, 1u, 2u);
    var index = m.x | m.y | m.z;
    let comp = getMipByte(mip, index);
    return comp != 0u;
}
fn mip2(march: vec3<f32>) -> bool {
    var mip = get_mip1(march);
    return any(mip > 0u);
}
fn mip3(march: vec3<f32>) -> bool {
    var mip = get_mip2(march);
    var marchu = vec3<i32>(march);
    var mod4 = marchu % 32;
    var m = vec3<u32>(mod4 > 15) << vec3(0u, 1u, 2u);
    var index = m.x | m.y | m.z;
    let comp = getMipByte(mip, index);
    var mod2 = marchu % 16;
    m = vec3<u32>(mod2 > 7) << vec3(0u, 1u, 2u);
    index = m.x | m.y | m.z;
    let voxel = getMipBit(comp, index);
    return voxel != 0u;
}
fn mip4(march: vec3<f32>) -> bool {
    var mip = get_mip2(march);
    var marchu = vec3<i32>(march);
    var mod4 = marchu % 32;
    var m = vec3<u32>(mod4 > 15) << vec3(0u, 1u, 2u);
    var index = m.x | m.y | m.z;
    let comp = getMipByte(mip, index);
    return comp != 0u;
}
fn mip5(march: vec3<f32>) -> bool {
    var mip = get_mip2(march);
    return any(mip > 0u);
}

fn get_prepass_depth(uv: vec2<f32>) -> f32 {
    // let index = vec2<i32>(uv * vec2(853.0, 480.0));
    let index = vec2<i32>(uv * vec2(640.0, 360.0));
    var t = textureLoad(depth_texture, index, 0).x;
    if true {
        // return t;
    }
    t = min(t, textureLoad(depth_texture, index + vec2(-1, 0), 0).x);
    t = min(t, textureLoad(depth_texture, index + vec2(1, 0), 0).x);
    t = min(t, textureLoad(depth_texture, index + vec2(0, -1), 0).x);
    t = min(t, textureLoad(depth_texture, index + vec2(0, 1), 0).x);
    t = min(t, textureLoad(depth_texture, index + vec2(1, 1), 0).x);
    t = min(t, textureLoad(depth_texture, index + vec2(-1, -1), 0).x);
    t = min(t, textureLoad(depth_texture, index + vec2(-1, 1), 0).x);
    t = min(t, textureLoad(depth_texture, index + vec2(1, -1), 0).x);
    return t;
}

// fn resolution_is_enouugh(t: f32) -> bool

struct Output {
    @location(0) color: vec4<f32>,
    #ifdef CHUNK_DEPTH_PREPASS
        @builtin(frag_depth) depth: f32,
    #endif
}

const nudge_t: f32 = 0.0005;
const enable_depth_prepass: bool = true;
// const enable_depth_prepass: bool = false;

// https://www.shadertoy.com/view/4dX3zl
// http://www.cse.yorku.ca/~amana/research/grid.pdf
// could also explore sphere/cube assisted marching
// - [Raymarching Voxel](https://medium.com/@calebleak/raymarching-voxel-rendering-58018201d9d6)
// - [VertexOutput](https://github.com/bevyengine/bevy/blob/71adb77a2ea97027ae54dea5552ba9fdbfb707fd/crates/bevy_pbr/src/render/forward_io.wgsl#L30)
@fragment
fn fragment(
    mesh: VertexOutput,
) -> Output {
    var out: Output;

    if (true) {
        // return vec4<f32>((mesh.world_position.xyz - chunk_pos)/1000.0, 1.0);
        // return vec4<f32>(mesh.uv, 0.0, 1.0);
    }
    let screen_uv = mesh.position.xy/vec2<f32>(world_data.screen_resolution.xy);
    let ray_origin = world_data.player_pos.xyz;
    let backface_hit_pos = mesh.world_position.xyz;
    let ray_dir = normalize(backface_hit_pos - ray_origin);
    let voxel_size = chunk_size * 2.0 / f32(side);
    let chunk_center = chunk_pos;

    #ifdef CHUNK_DEPTH_PREPASS
        if !enable_depth_prepass {
            out.color = vec4(0.0);
            out.depth = 1.0;
            return out;
        }
    #else
        let depth_t = get_prepass_depth(screen_uv);
        if length(backface_hit_pos - ray_origin) < depth_t {
            discard;
        }
    #endif

    // calculate ray hit pos
    
    var o = ray_origin;
    // how much t for 1 voxel unit in this direction
    var dt = (1.0 / ray_dir);

    // https://gamedev.stackexchange.com/a/18459
    // we know the ray hits, so we skip the 2 checks for ray not hitting the box
    let v1 = chunk_center - chunk_size;
    let v2 = chunk_center + chunk_size;
    let t1 = (v1 - o)*dt;
    let t2 = (v2 - o)*dt;
    let tmin = min(t1, t2);

    // if t is -ve, we want to march from the ray_origin instead. so clip it at 0.0
    let t = max(0.0, max(tmin.x, max(tmin.y, tmin.z)));

    #ifdef CHUNK_DEPTH_PREPASS
        o = ray_origin + ray_dir * t;
    #else
    if enable_depth_prepass {
        o = ray_origin + ray_dir * (max(depth_t - 0.000, t));
        // o = ray_origin + ray_dir * (depth_t + 0.000);

        // out.color = vec4(vec3(depth_t/10.0), 1.0);
        // out.color = vec4(vec3(50.0/t), 1.0);
        // out.color = vec4(vec3(depth_t - t)/100.0, 1.0);
        // out.color = vec4(vec3(max(depth_t, t))/1000.0, 1.0);
        // out.color = vec4(vec3(depth_t)/1000.0, 1.0);
        // out.color = vec4(vec3(50.0/depth_t), 1.0);
        // return out;
    }
    #endif

    // cache the hit position
    let ray_pos = o;


    // marching inside the chunk

    // we only want the magnitude of dt for marching
    dt = abs(dt);
    // origin wrt chunk's origin
    o -= (chunk_center - chunk_size);
    // make each voxel of size 1
    o /= voxel_size;

    var stepi = vec3<i32>(ray_dir >= 0.0) - vec3<i32>(ray_dir < 0.0);
    var step = vec3<f32>(stepi);
    #ifdef CHUNK_DEPTH_PREPASS
        var res = mip5_loop_final(o, ray_dir, step, stepi, dt);
    #else
        var res = mip2_loop_final(o, ray_dir, step, stepi, dt);
    #endif
    // var res = inline_mip2_loop(o, ray_dir, step, stepi, dt);
    // var res = inline_no_mip_loop(o, ray_dir, step, dt);
    res.t *= voxel_size;
    res.t += t;

    #ifdef CHUNK_DEPTH_PREPASS
        out.color = vec4(vec3(res.t), 1.0);
        out.depth = 1.0/res.t;
    #else
        out.color = res.color;
    #endif

    if res.hit {
        return out;
    }

    #ifdef CHUNK_DEPTH_PREPASS
        out.color = vec4(100000000.0);
        out.depth = 0.000001;
        return out;
    #else
        discard;
    #endif
}

struct MipResult {
    color: vec4<f32>,
    hit: bool,
    t: f32,
}

fn inline_no_mip_loop(_o: vec3<f32>, ray_dir: vec3<f32>, step: vec3<f32>, dt: vec3<f32>) -> MipResult {
    var res: MipResult;
    res.hit = false;
    res.color = vec4(0.0);

    var o = _o + ray_dir * 0.001;

    // how much t untill we hit a plane along this axis
    var t = (step * 0.5 + 0.5 - fract(o) * step) * dt;
    // current voxel position (offset to center of the voxel)
    var march = floor(o) + 0.5;

    var voxel: u32;
    var mask: vec3<f32>;
    for (var i = 0u; i < side * 3u - 2u; i += 1u) {
        voxel = get_voxel(march);
        if (voxel != 0u) {
            res.color = get_color(voxel);
            res.hit = true;
            return res;
        }

        mask = vec3<f32>(t.xyz <= min(t.yzx, t.zxy));
        t += mask * dt;
        march += mask * step;
    }

    return res;
}

fn inline_mip2_loop(_o: vec3<f32>, ray_dir: vec3<f32>, step: vec3<f32>, stepi: vec3<i32>, dt: vec3<f32>) -> MipResult {
    var res: MipResult;
    res.hit = false;
    res.color = vec4(0.0);

    var o = _o + ray_dir * 0.001;
    // current voxel position (offset to center of the voxel)
    var march = floor(o) + 0.5;
    var marchi = vec3<i32>(march);

    // how much t untill we hit a plane along this axis
    var t = (step * (0.5 - fract(o/4.0)) + 0.5) * dt * 4.0;
    var voxel: u32;
    var mask: vec3<f32>;
    var last_t = 0.0;
    for (var i = 0u; i < (side / 4u) * 3u - 2u; i += 1u) {
        if any(marchi >= i32(side) || marchi < 0) {
            break;
        }
        var mip = get_mip1(march);
        if any(mip > 0u) {
            var o = o + ray_dir * (last_t + 0.001);
            var march = floor(o) + 0.5;
            var marchi = vec3<i32>(march);
            var t = (step * (0.5 - fract(o/2.0)) + 0.5) * dt * 2.0;
            var mod4 = marchi % 4;
            var last_t = 0.0;
            for (var i2 = 0u; i2 < 4u; i2 += 1u) {
                if (any(mod4 >= 4 || mod4 < 0)) {
                    break;
                }
                let m = vec3<u32>(mod4 > 1) << vec3(0u, 1u, 2u);
                let index = m.x | m.y | m.z;
                let comp = getMipByte(mip, index);
                if (comp > 0u) {
                    var o = o + ray_dir * (last_t + 0.001);
                    var march = floor(o) + 0.5;
                    var marchi = vec3<i32>(march);
                    var t = (step * (0.5 - fract(o)) + 0.5) * dt;
                    var mod2 = marchi % 2;
                    var last_t = 0.0;
                    for (var i3 = 0u; i3 < 4u; i3 += 1u) {
                        if any(mod2 >= 2 || mod2 < 0) {
                            break;
                        }
                        let m = vec3<u32>(mod2 > 0) << vec3(0u, 1u, 2u);
                        let index = m.x | m.y | m.z;
                        let voxel = getMipBit(comp, index);
                        if voxel > 0u {
                            let voxel = get_voxel(march);
                            // check again cux of precision issues ig :/
                            if voxel > 0u {
                                res.color = get_color(voxel);
                                res.hit = true;
                                return res;
                            }
                        }
                        let maski = vec3<i32>(t.xyz <= min(t.yzx, t.zxy));
                        let mask = vec3<f32>(maski);
                        last_t = min(t.x, min(t.y, t.z));
                        t += mask * dt;
                        march += mask * step;
                        mod2 += maski * stepi;
                    }
                }
                let maski = vec3<i32>(t.xyz <= min(t.yzx, t.zxy));
                let mask = vec3<f32>(maski);
                last_t = min(t.x, min(t.y, t.z));
                t += mask * dt * 2.0;
                march += mask * step * 2.0;
                mod4 += maski * stepi * 2;
            }
        }
        let maski = vec3<i32>(t.xyz <= min(t.yzx, t.zxy));
        let mask = vec3<f32>(maski);
        last_t = min(t.x, min(t.y, t.z));
        t += mask * dt * 4.0;
        march += mask * step * 4.0;
        marchi += maski * stepi * 4;
    }

    return res;
}

fn mip5_loop_final(_o: vec3<f32>, ray_dir: vec3<f32>, step: vec3<f32>, _stepi: vec3<i32>, dt: vec3<f32>) -> MipResult {
    var res: MipResult;
    res.hit = false;
    res.color = vec4(0.0);

    var o = _o + ray_dir * nudge_t;
    // current voxel position (offset to center of the voxel)
    var marchi = vec3<i32>(floor(o) + 0.5);

    let stepi = _stepi * 32;
    o = o / 32.0;
    let lim = i32(side / 32u);

    // how much t untill we hit a plane along this axis
    var t = (step * (0.5 - fract(o)) + 0.5) * dt;
    var last_t = 0.0;
    for (var i = 0; i < lim * 3 - 2; i += 1) {
        // TODO: figure out why this should not be '>='
        if any(marchi > i32(side) || marchi < 0) {
            break;
        }
        var mip = get_mip2_unckecked(marchi);
        if any(mip > 0u) {
            let res = mip4_loop(_o, ray_dir, step, _stepi, dt, last_t * 32.0 + nudge_t, mip);
            if res.hit {
                return res;
            }
        }
        let maski = vec3<i32>(t.xyz <= min(t.yzx, t.zxy));
        last_t = min(t.x, min(t.y, t.z));
        t += vec3<f32>(maski) * dt;
        marchi += maski * stepi;
    }
    
    return res;
}

fn mip4_loop(_o: vec3<f32>, ray_dir: vec3<f32>, step: vec3<f32>, _stepi: vec3<i32>, dt: vec3<f32>, _last_t: f32, mip: vec2<u32>) -> MipResult {
    var res: MipResult;
    res.hit = false;
    res.color = vec4(0.0);

    var o = _o + ray_dir * (_last_t + nudge_t);

    var mod32 = vec3<i32>(floor(o) + 0.5) % 32;
    o = o / 16.0;
    let stepi = _stepi * 16;

    var t = (step * (0.5 - fract(o)) + 0.5) * dt;
    var last_t = 0.0;
    for (var i = 0; i < 4; i += 1) {
        if (any(mod32 >= 32 || mod32 < 0)) {
            break;
        }
        let m = vec3<u32>(mod32 > 15) << vec3(0u, 1u, 2u);
        let index = m.x | m.y | m.z;
        let comp = getMipByte(mip, index);
        if (comp > 0u) {
            res = mip3_loop(_o, ray_dir, step, _stepi, dt, _last_t + last_t * 16.0 + nudge_t, comp);
            if res.hit {
                return res;
            }
        }
        let maski = vec3<i32>(t.xyz <= min(t.yzx, t.zxy));
        last_t = min(t.x, min(t.y, t.z));
        t += vec3<f32>(maski) * dt;
        mod32 += maski * stepi;
    }
    
    return res;
}

fn mip3_loop(_o: vec3<f32>, ray_dir: vec3<f32>, step: vec3<f32>, _stepi: vec3<i32>, dt: vec3<f32>, _last_t: f32, comp: u32) -> MipResult {
    var res: MipResult;
    res.hit = false;
    res.color = vec4(0.0);

    var o = _o + ray_dir * (_last_t + nudge_t);

    var mod16 = vec3<i32>(floor(o) + 0.5) % 16;
    o = o / 8.0;
    let stepi = _stepi * 8;

    var t = (step * (0.5 - fract(o)) + 0.5) * dt;
    var last_t = 0.0;
    for (var i = 0; i < 4; i += 1) {
        if any(mod16 >= 16 || mod16 < 0) {
            break;
        }
        let m = vec3<u32>(mod16 > 7) << vec3(0u, 1u, 2u);
        let index = m.x | m.y | m.z;
        let voxel = getMipBit(comp, index);
        if voxel > 0u {
            let res = mip2_loop(_o, ray_dir, step, _stepi, dt, _last_t + last_t * 8.0 + nudge_t);
            if res.hit {
                return res;
            }
        }
        let maski = vec3<i32>(t.xyz <= min(t.yzx, t.zxy));
        last_t = min(t.x, min(t.y, t.z));
        t += vec3<f32>(maski) * dt;
        mod16 += maski * stepi;
    }

    return res;
}

fn mip2_loop_final(_o: vec3<f32>, ray_dir: vec3<f32>, step: vec3<f32>, stepi: vec3<i32>, dt: vec3<f32>) -> MipResult {
    var res: MipResult;
    res.hit = false;
    res.color = vec4(0.0);

    var o = _o + ray_dir * nudge_t;
    // current voxel position (offset to center of the voxel)
    var marchi = vec3<i32>(floor(o) + 0.5);

    // how much t untill we hit a plane along this axis
    var t = (step * (0.5 - fract(o/4.0)) + 0.5) * dt * 4.0;
    var last_t = 0.0;
    for (var i = 0u; i < (side / 4u) * 3u - 2u; i += 1u) {
        if any(marchi >= i32(side) || marchi < 0) {
            break;
        }
        var mip = get_mip1_unchecked(marchi);
        if any(mip > 0u) {
            let res = mip1_loop(o, ray_dir, step, stepi, dt, last_t + nudge_t, mip);
            if res.hit {
                return res;
            }
        }
        let maski = vec3<i32>(t.xyz <= min(t.yzx, t.zxy));
        let mask = vec3<f32>(maski);
        last_t = min(t.x, min(t.y, t.z));
        t += mask * dt * 4.0;
        marchi += maski * stepi * 4;
    }
    
    return res;
}

fn mip2_loop(_o: vec3<f32>, ray_dir: vec3<f32>, step: vec3<f32>, _stepi: vec3<i32>, dt: vec3<f32>, _last_t: f32) -> MipResult {
    var res: MipResult;
    res.hit = false;
    res.color = vec4(0.0);

    var o = _o + ray_dir * (_last_t + nudge_t);
    var marchi = vec3<i32>(floor(o) + 0.5);

    var mod8 = marchi % 8;
    marchi -= mod8;
    o /= 4.0;
    let stepi = _stepi * 4;

    var t = (step * (0.5 - fract(o)) + 0.5) * dt;
    var last_t = 0.0;
    for (var i = 0; i < 4; i += 1) {
        if (any(mod8 >= 8 || mod8 < 0)) {
            break;
        }
        var mip = get_mip1_unchecked(marchi + mod8);
        if any(mip > 0u) {
            let res = mip1_loop(_o, ray_dir, step, _stepi, dt, _last_t + last_t * 4.0 + nudge_t, mip);
            if res.hit {
                return res;
            }
        }
        let maski = vec3<i32>(t.xyz <= min(t.yzx, t.zxy));
        last_t = min(t.x, min(t.y, t.z));
        t += vec3<f32>(maski) * dt;
        mod8 += maski * stepi;
    }
    
    return res;
}

fn mip1_loop(_o: vec3<f32>, ray_dir: vec3<f32>, step: vec3<f32>, _stepi: vec3<i32>, dt: vec3<f32>, _last_t: f32, mip: vec2<u32>) -> MipResult {
    var res: MipResult;
    res.hit = false;
    res.color = vec4(0.0);

    var o = _o + ray_dir * (_last_t + nudge_t);

    var mod4 = vec3<i32>(floor(o) + 0.5) % 4;
    o /= 2.0;
    let stepi = _stepi * 2;

    var t = (step * (0.5 - fract(o)) + 0.5) * dt;
    var last_t = 0.0;
    for (var i = 0; i < 4; i += 1) {
        if (any(mod4 >= 4 || mod4 < 0)) {
            break;
        }
        let m = vec3<u32>(mod4 > 1) << vec3(0u, 1u, 2u);
        let index = m.x | m.y | m.z;
        let comp = getMipByte(mip, index);
        if (comp > 0u) {
            res = mip0_loop(_o, ray_dir, step, _stepi, dt, _last_t + last_t * 2.0 + nudge_t, comp);
            if res.hit {
                return res;
            }
        }
        let maski = vec3<i32>(t.xyz <= min(t.yzx, t.zxy));
        last_t = min(t.x, min(t.y, t.z));
        t += vec3<f32>(maski) * dt;
        mod4 += maski * stepi;
    }
    
    return res;
}

fn mip0_loop(_o: vec3<f32>, ray_dir: vec3<f32>, step: vec3<f32>, stepi: vec3<i32>, dt: vec3<f32>, _last_t: f32, comp: u32) -> MipResult {
    var res: MipResult;
    res.hit = false;
    res.color = vec4(0.0);

    var o = _o + ray_dir * (_last_t + nudge_t);
    var marchi = vec3<i32>(floor(o) + 0.5);

    var mod2 = marchi % 2;
    marchi -= mod2;

    var t = (step * (0.5 - fract(o)) + 0.5) * dt;
    var last_t = 0.0;
    for (var i = 0; i < 4; i += 1) {
        if any(mod2 >= 2 || mod2 < 0) {
            break;
        }
        let m = vec3<u32>(mod2 > 0) << vec3(0u, 1u, 2u);
        let index = m.x | m.y | m.z;
        let voxel = getMipBit(comp, index);
        if voxel > 0u {
            #ifdef CHUNK_DEPTH_PREPASS
                res.color = vec4(vec3(_last_t + last_t), 1.0);
                res.hit = true;
                res.t = _last_t + last_t;
                return res;
            #else
                let voxel = get_voxel_unchecked(marchi + mod2);
                // check again cux of precision issues ig :/
                if voxel > 0u {
                    res.color = get_color(voxel);
                    res.hit = true;
                    res.t = _last_t + last_t;
                    return res;
                }
            #endif
        }
        let maski = vec3<i32>(t.xyz <= min(t.yzx, t.zxy));
        last_t = min(t.x, min(t.y, t.z));
        t += vec3<f32>(maski) * dt;
        mod2 += maski * stepi;
    }

    return res;
}
