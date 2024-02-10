#import bevy_pbr::forward_io::VertexOutput

@group(1) @binding(0) var<uniform> side: u32;
@group(1) @binding(5) var<uniform> chunk_pos: vec3<f32>;
@group(1) @binding(6) var<uniform> voxel_size: f32;

@group(1) @binding(1) var voxels: texture_3d<u32>;
@group(1) @binding(7) var voxels_mip1: texture_3d<u32>;
@group(1) @binding(8) var voxels_mip2: texture_3d<u32>;
@group(1) @binding(2) var materials: texture_1d<f32>;

struct WorldDataUniforms {
    player_pos: vec3<f32>,
    screen_resolution: vec2<u32>,
    fov: f32,
    unit_t_max: f32,
    depth_prepass_scale_factor: u32,
}

@group(3) @binding(0) var<uniform> world_data: WorldDataUniforms;
@group(3) @binding(1) var transient_world_textre: texture_3d<u32>;
@group(3) @binding(2) var depth_texture: texture_2d<f32>;

struct BitWorldUniforms {
    chunk_side: u32,
    world_side: u32,
    pos: vec3<f32>,
}
@group(3) @binding(3) var<uniform> bit_world_uniforms: BitWorldUniforms;
@group(3) @binding(4) var<storage> bit_world_chunk_buffer: array<vec2<u32>>;
@group(3) @binding(5) var<storage> bit_world_chunk_indices: array<u32>;

// returns voxel material in the rightmost byte
fn get_voxel(pos: vec3<f32>) -> u32 {
    if (any(pos >= f32(side) || pos < 0.0)) {
        return 0u;
    }
    
    // loading textures automatically break the u32 into a vec4<u32> 1 byte each channel (~~ vec4<u8>)
    var voxel = textureLoad(voxels, vec3<i32>(pos), 0).x;

    return voxel;
}

fn get_color(index: u32) -> vec4<f32> {
    return textureLoad(materials, index, 0);
}


fn unpackBytes(t: u32) -> vec4<u32> {
    return (t >> vec4<u32>(0u, 8u, 16u, 24u)) & 15u;
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

fn get_prepass_depth(uv: vec2<f32>) -> f32 {
    let index = vec2<i32>(
        uv *
        (vec2<f32>(world_data.screen_resolution / world_data.depth_prepass_scale_factor) +
        vec2<f32>(world_data.screen_resolution % world_data.depth_prepass_scale_factor > 0u))
    );
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

fn is_resolution_enough(t: f32, voxel_size: f32) -> bool {
    #ifdef CHUNK_DEPTH_PREPASS
        return world_data.unit_t_max * voxel_size > t;
    #else
        return world_data.unit_t_max * voxel_size * f32(world_data.depth_prepass_scale_factor) > t;
    #endif
}

struct Output {
    @location(0) color: vec4<f32>,
    #ifdef CHUNK_DEPTH_PREPASS
        @builtin(frag_depth) depth: f32,
    #endif
}

// const enable_depth_prepass: bool = true;
const enable_depth_prepass: bool = false;

// https://www.shadertoy.com/view/4dX3zl
// http://www.cse.yorku.ca/~amana/research/grid.pdf
// could also explore sphere/cube assisted marching
// - [Raymarching Voxel](https://medium.com/@calebleak/raymarching-voxel-rendering-58018201d9d6)
// - [VertexOutput](https://github.com/bevyengine/bevy/blob/71adb77a2ea97027ae54dea5552ba9fdbfb707fd/crates/bevy_pbr/src/render/forward_io.wgsl#L30)
@fragment
fn fragment(mesh: VertexOutput) -> Output {
    var out: Output;

    if true {
        // out.color =  vec4<f32>((mesh.world_position.xyz - chunk_pos)/1000.0, 1.0);
        // out.color = vec4<f32>(mesh.uv, 0.0, 1.0);
        // out.color = vec4<f32>(abs(chunk_pos.xz - world_data.player_pos.xz)/100.0, 0.0, 0.0);
        // out.color = vec4(chunk_pos.xz - mesh.world_position.xz, 0.0, 0.0);
        // return out;

        // #ifdef CHUNK_RENDER_PASS
        // let screen_uv = mesh.position.xy/vec2<f32>(world_data.screen_resolution.xy);
        // let t = 256.0 * 4.0;
        // let uv = floor(screen_uv * t);
        // let index = uv.y * t + uv.x;
        // out.color = vec4(f32(bit_world_chunk_indices[i32(index)] > 0u));
        // // out.color = vec4(uv/256.0, 1.0, 1.0);
        // // out.color = vec4(index / 100000.0);
        // return out;
        // #endif
    }
    let screen_uv = mesh.position.xy/vec2<f32>(world_data.screen_resolution.xy);
    let ray_origin = world_data.player_pos.xyz;
    let backface_hit_pos = mesh.world_position.xyz;
    let ray_dir = normalize(backface_hit_pos - ray_origin);
    let chunk_size = f32(side) * voxel_size / 2.0;
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
    var t = max(0.0, max(tmin.x, max(tmin.y, tmin.z)));

    #ifdef CHUNK_DEPTH_PREPASS
    #else
    if enable_depth_prepass {
        t = max(t, depth_t);

        // out.color = vec4(vec3(50.0/t), 1.0);
        // out.color = vec4(vec3(depth_t - t)/100.0, 1.0);
        // out.color = vec4(vec3(max(depth_t, t))/1000.0, 1.0);
        // out.color = vec4(vec3(depth_t)/1000.0, 1.0);
        // out.color = vec4(vec3(50.0/depth_t), 1.0);
        // return out;
    }
    #endif
    o = ray_origin + ray_dir * t;

    // cache the hit position
    let ray_pos = o;


    // marching inside the chunk

    // we only want the magnitude of dt for marching
    dt = abs(dt);
    // origin wrt chunk's origin
    o -= (chunk_center - chunk_size);
    // make each voxel of size 1
    o /= voxel_size;

    // TODO: could try to rewrite this to make ray direction always be >= 0 in every direction
    // and store an (offset, sign) pair to calculate the voxel position
    // march * sign + offset
    // offset.x is 'side' and sign.x is -1 if ray goes towards -ve x for example.
    // after this, it becomes much easier to do the 2 component thing i think.

    var stepi = vec3<i32>(ray_dir >= 0.0) - vec3<i32>(ray_dir < 0.0);
    var step = vec3<f32>(stepi);
    var res: MipResult;
    #ifdef CHUNK_DEPTH_PREPASS
        res = mip5_loop_final(o, ray_dir, step, stepi, dt, t / voxel_size);
        // res = mip2_loop_final(o, ray_dir, step, stepi, dt, t / voxel_size);
    #else
        if enable_depth_prepass {
            res = mip2_loop_final(o, ray_dir, step, stepi, dt, t / voxel_size);
            // res = inline_no_mip_loop(o, ray_dir, step, dt);
            // res = bitworld_trace(ray_pos, ray_dir, step, dt);
        } else {
            // res = inline_no_mip_loop(o, ray_dir, step, dt);
            // res = mip2_loop_final(o, ray_dir, step, stepi, dt, t / voxel_size);
            // res = mip5_loop_final(o, ray_dir, step, stepi, dt, t / voxel_size);
            res = trace_while(o, ray_dir, stepi, dt, t / voxel_size);
            // res = bitworld_trace(ray_pos, ray_dir, step, dt);
        }
    #endif
    res.t *= voxel_size;
    res.t += t;

    #ifdef CHUNK_DEPTH_PREPASS
        out.color = vec4(vec3(res.t), 1.0);
        out.depth = 1.0/res.t;
    #else
        out.color = res.color;
    #endif

    if res.hit {
        #ifdef CHUNK_RENDER_PASS
            // let dir = normalize(vec3(1.0, 1.0, 1.0));
            // let pos = ray_origin + ray_dir * res.t * 1.0 + dir * 1.0;
            // // out.color = vec4(pos/200.0, 1.0);
            // // out.color = vec4(vec3(res.t/300.0), 1.0);
            // var sun_res = bitworld_trace(pos, dir, step, dt);
            // if sun_res.hit {
            //     out.color *= 0.5;
            // }
        #endif
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

// given that side is 2^_ and k is 2^_
// k
//  - final 5
//    - 2 for 0, 1, 2, 3, 4
//    - side / 32
//  - final 2
//    - 2 for 0, 1
//    - side / 4
// outmask
//  - final 5
//    - !1 for 0, 1, 2, 3, 4
//    - !(side/32 - 1) for 5
//  - final 2
//    - !1 for 0, 1
//    - !(side/4 - 1) for 2
struct RayMarch {
    o: vec3<f32>,
    march: vec3<i32>,
    modk: vec3<i32>,
    t: vec3<f32>,
    data: vec2<u32>,
    outmask: i32,
    // k: i32,
    last_t: f32,
}
struct Ray {
    // origin: vec3<f32>,
    dir: vec3<f32>,
    dt: vec3<f32>,
    step: vec3<i32>,
    // mip: u32,
    // index: u32,
    max_index: i32,
    stack: array<RayMarch, 10>,
}

fn trace_while(o: vec3<f32>, dir: vec3<f32>, step: vec3<i32>, dt: vec3<f32>, ray_t: f32) -> MipResult {
    var res = MipResult();
    var r = Ray();
    r.dir = dir;
    r.dt = dt;
    r.step = step;
    r.max_index = 2;
    r.max_index = 5;
    let k = i32(1u << u32(r.max_index));
    var rm = RayMarch();
    rm.o = o/f32(k);
    rm.outmask = !(i32(side)/k - 1);
    rm.march = vec3<i32>(rm.o);
    rm.march = max(vec3(0), rm.march);
    rm.march = min(vec3(i32(side - 1u)/k), rm.march);
    rm.t = (vec3<f32>(r.step) * (0.5 - (rm.o - vec3<f32>(rm.march))) + 0.5) * r.dt;
    rm.modk = rm.march & !rm.outmask; // &!outmask or &(rm.k - 1) or %rm.k
    rm.march -= rm.modk;
    var index = r.max_index;

    // if any(rm.march != vec3(0)) {
    //     res.hit = true;
    //     res.color = vec4(0.0, 1.0, 0.0, 1.0);
    //     return res;
    // }

    let too_deep = 500;
    // while index >= 0 && index <= r.max_index {
    for (var i=0; i<too_deep && index >= 0 && index <= r.max_index; i += 1) {
        // if i == too_deep - 1 {
        //     res.hit = true;
        //     res.color = vec4(1.0, 0.0, 1.0, 1.0);
        //     return res;
        // }
        if any((rm.modk & rm.outmask) != 0) {
        // if any(rm.modk >= i32(side)/32 || rm.modk < 0) {
            index += 1;
            rm = r.stack[index];

            let mask = vec3<i32>(rm.t.xyz <= min(rm.t.yzx, rm.t.zxy));
            rm.last_t = min(rm.t.x, min(rm.t.y, rm.t.z));
            rm.t += vec3<f32>(mask) * r.dt;
            rm.modk += mask * r.step;
            continue;
        }
        var mip = vec2<u32>();
        switch index {
            case 0, 3: {
                let m = vec3<u32>(rm.modk > 0) << vec3(0u, 1u, 2u);
                let mipi = m.x | m.y | m.z;
                mip = vec2(getMipBit(rm.data.x, mipi), 0u);
            }
            case 1, 4: {
                let m = vec3<u32>(rm.modk > 0) << vec3(0u, 1u, 2u);
                let mipi = m.x | m.y | m.z;
                mip = vec2(getMipByte(rm.data, mipi), 0u);
            }
            case 2: {
                mip = textureLoad(voxels_mip1, rm.march + rm.modk, 0).xy;
            }
            case 5: {
                mip = textureLoad(voxels_mip2, rm.march + rm.modk, 0).xy;
            }
            default: {
                break;
            }
        }
        if any(mip > 0u) {
            // if index == 6 {
            //     res.hit = true;
            //     res.color = vec4(vec3<f32>(rm.march + rm.modk)/(10.0 * f32(6 - index)), 1.0);
            //     // res.color = vec4(1.0/vec3(rm.last_t * f32(1 << u32(index)) + ray_t)/0000.003, 1.0);
            //     // res.color = vec4(o/200.0, 1.0);
            //     // res.color = vec4(vec3(rm.last_t/10.0), 1.0);
            //     return res;
            // }

            r.stack[index] = rm;
            index -= 1;

            rm.o += r.dir * rm.last_t;
            rm.o *= 2.0;
            rm.last_t = 0.0;
            let min_bound = (rm.march + rm.modk) * 2;
            rm.march = vec3<i32>(rm.o);
            rm.march = min(min_bound + 1, rm.march);
            rm.march = max(min_bound, rm.march);
            rm.t = (vec3<f32>(r.step) * (0.5 - (rm.o - vec3<f32>(rm.march))) + 0.5) * r.dt;

            switch index | i32(r.max_index == 5) * 32 {
                case -1: {break;}
                case 0, 1, 32, 33, 34, 35, 36: {
                    rm.outmask = !(2 - 1);
                }
                case 2: {
                    rm.outmask = !(i32(side)/4 - 1);
                }
                case 37 {
                    rm.outmask = !(i32(side)/32 - 1);
                }
                default: {
                    break;
                }
            }

            rm.modk = rm.march & !rm.outmask; // &!outmask or &(rm.k - 1) or %rm.k
            rm.march -= rm.modk;
            rm.data = mip;
            continue;
        }
        let mask = vec3<i32>(rm.t.xyz <= min(rm.t.yzx, rm.t.zxy));
        rm.last_t = min(rm.t.x, min(rm.t.y, rm.t.z));
        rm.t += vec3<f32>(mask) * r.dt;
        rm.modk += mask * r.step;
    }

    if index == -1 {
        rm = r.stack[0];
        res.hit = true;
        res.t = rm.last_t;
        let voxel = textureLoad(voxels, rm.march + rm.modk, 0).x;
        res.color = get_color(voxel);
    }
    return res;
}

fn bitworld_trace(_o: vec3<f32>, ray_dir: vec3<f32>, step: vec3<f32>, dt: vec3<f32>) -> MipResult {
    var res: MipResult;
    res.hit = false;
    res.color = vec4(0.0);

    let su = i32(bit_world_uniforms.world_side);
    let s = f32(su);
    // if bit_world_uniforms.world_side == 32u {
    //     res.color = vec4(bit_world_uniforms.pos, 1.0);
    //     res.hit = true;
    //     return res;
    // }
    
    // let chunk_mip = bit_world_chunk_indices;
    var o = (_o - bit_world_uniforms.pos) + ray_dir * 0.001;
    o /= voxel_size;
    o += s / 2.0 * f32(bit_world_uniforms.chunk_side);
    // if true {
    //     res.hit = true;
    //     let marchi = vec3<i32>(o)/128;
    //     let index = marchi.z * su * su + marchi.y * su + marchi.x;
    //     // res.color = vec4(vec3<f32>(marchi)/(32.0 * 1.0), 1.0);
    //     res.color = vec4(vec3(f32(bit_world_chunk_indices[index] > 0u)), 1.0);
    //     // res.color = vec4(o/100000.0, 1.0);
    //     return res;
    // }

    let sw = i32(bit_world_uniforms.world_side);
    let sc = i32(bit_world_uniforms.chunk_side);
    for (var i = 0; i < 64; i += 1) {
        if any(o > f32(sw * sc)) || any(o < 0.0) {
            break;
        }
        let marchi = vec3<i32>(o)/sc;
        let index = marchi.z * sw * sw + marchi.y * sw + marchi.x;
        let chunk_index = bit_world_chunk_indices[index];
        if chunk_index > 0u {
            let v_pos = vec3<i32>(o) - marchi * sc;
            // if true {
            //     res.color = vec4(vec3<f32>(v_pos), 1.0);
            //     res.hit = true;
            //     return res;
            // }
            let marchi = v_pos / 4;
            let sc = sc / 4;
            let index = marchi.z * sc * sc + marchi.y * sc + marchi.x;
            let voxel = bit_world_chunk_buffer[chunk_index + u32(index)];
            if any(voxel > 0u) {
                // res.color = vec4(1.0);
                // res.color = get_color(get_voxel(vec3<f32>(v_pos)));
                res.hit = true;
                return res;
            }
        }

        o += ray_dir * 4.0;
    }

    // // how much t untill we hit a plane along this axis
    // var t = (step * 0.5 + 0.5 - fract(o) * step) * dt;
    // // current voxel position (offset to center of the voxel)
    // var march = floor(o) + 0.5;

    // var voxel: u32;
    // var mask: vec3<f32>;
    // for (var i = 0u; i < side * 3u - 2u; i += 1u) {
    //     let marchi = vec3<i32>(march);
    //     let index = marchi.z * s * s + marchi.y * s + marchi.x;

    //     // voxel = chunk_mip[index];
    //     if (voxel != 0u) {
    //         res.color = vec4(0.8);
    //         res.hit = true;
    //         return res;
    //     }

    //     mask = vec3<f32>(t.xyz <= min(t.yzx, t.zxy));
    //     t += mask * dt;
    //     march += mask * step;
    // }

    return res;
}

fn trace_sun() -> vec3<f32> {
    return vec3<f32>(0.0);
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

fn mip5_loop_final(_oo: vec3<f32>, ray_dir: vec3<f32>, step: vec3<f32>, stepi: vec3<i32>, dt: vec3<f32>, ray_t: f32) -> MipResult {
    var res: MipResult;
    res.hit = false;
    res.color = vec4(0.0);

    var _o = _oo / 32.0;
    var o = _o;
    // current voxel position
    var marchi = vec3<i32>(o);

    // this is kinda like having 2 components of ray_pos. an integer component and a fractional component.
    // when intersecting ray with a voxel, we already know
    // the bounds where the voxel could be
    // as it should be within the octree node that is hit previously.
    // so after intersection, we just do
    // march = max(min(march, max_bound), min_bound)
    marchi = max(vec3(0), marchi);
    marchi = min(vec3(i32(side - 1u)/32), marchi);

    // how much t untill we hit a plane along this axis
    var t = (step * (0.5 - (o - vec3<f32>(marchi))) + 0.5) * dt;
    var last_t = 0.0;
    for (var i = 0; i < i32(side)/32 * 3 - 2; i += 1) {
        if any(marchi >= i32(side)/32 || marchi < 0) {
            break;
        }
        var mip = textureLoad(voxels_mip2, marchi, 0).xy;
        if any(mip > 0u) {
            let res = mip4_loop(_o * 2.0, ray_dir, step, stepi, dt, last_t * 2.0, mip, ray_t, marchi * 2);
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

fn mip4_loop(_o: vec3<f32>, ray_dir: vec3<f32>, step: vec3<f32>, stepi: vec3<i32>, dt: vec3<f32>, _last_t: f32, mip: vec2<u32>, ray_t: f32, min_bound: vec3<i32>) -> MipResult {
    var res: MipResult;
    res.hit = false;
    res.color = vec4(0.0);

    var o = _o + ray_dir * _last_t;

    var marchi = vec3<i32>(o);
    marchi = min(min_bound + 1, marchi);
    marchi = max(min_bound, marchi);
    var t = (step * (0.5 - (o - vec3<f32>(marchi))) + 0.5) * dt;
    var mod32 = marchi % 2;
    marchi -= mod32;


    var last_t = 0.0;
    for (var i = 0; i < 4; i += 1) {
        if (any(mod32 >= 2 || mod32 < 0)) {
            break;
        }
        let m = vec3<u32>(mod32 > 0) << vec3(0u, 1u, 2u);
        let index = m.x | m.y | m.z;
        let comp = getMipByte(mip, index);
        if (comp > 0u) {
            res = mip3_loop(_o * 2.0, ray_dir, step, stepi, dt, (_last_t + last_t) * 2.0, comp, ray_t, (marchi + mod32) * 2);
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

fn mip3_loop(_o: vec3<f32>, ray_dir: vec3<f32>, step: vec3<f32>, stepi: vec3<i32>, dt: vec3<f32>, _last_t: f32, comp: u32, ray_t: f32, min_bound: vec3<i32>) -> MipResult {
    var res: MipResult;
    res.hit = false;
    res.color = vec4(0.0);

    var o = _o + ray_dir * _last_t;
    var marchi = vec3<i32>(floor(o) + 0.5);
    marchi = min(min_bound + 1, marchi);
    marchi = max(min_bound, marchi);
    var t = (step * (0.5 - (o - vec3<f32>(marchi))) + 0.5) * dt;
    var mod16 = marchi % 2;
    marchi -= mod16;

    var last_t = 0.0;
    for (var i = 0; i < 4; i += 1) {
        if any(mod16 >= 2 || mod16 < 0) {
            break;
        }
        let m = vec3<u32>(mod16 > 0) << vec3(0u, 1u, 2u);
        let index = m.x | m.y | m.z;
        let voxel = getMipBit(comp, index);
        if voxel > 0u {
            let res = mip2_loop(_o * 2.0, ray_dir, step, stepi, dt, (_last_t + last_t) * 2.0, ray_t, (marchi + mod16) * 2);
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

fn mip2_loop_final(_oo: vec3<f32>, ray_dir: vec3<f32>, step: vec3<f32>, stepi: vec3<i32>, dt: vec3<f32>, ray_t: f32) -> MipResult {
    var res: MipResult;
    res.hit = false;
    res.color = vec4(0.0);

    let _o = _oo / 4.0;
    var o = _o;
    var marchi = vec3<i32>(o);
    marchi = min(vec3(i32(side - 1u)/4), marchi);
    marchi = max(vec3(0), marchi);
    var t = (step * (0.5 - (o - vec3<f32>(marchi))) + 0.5) * dt;

    var last_t = 0.0;
    for (var i = 0u; i < (side / 4u) * 3u - 2u; i += 1u) {
        if any(marchi >= i32(side)/4 || marchi < 0) {
            break;
        }
        var mip = textureLoad(voxels_mip1, marchi, 0).xy;
        if any(mip > 0u) {
            let res = mip1_loop(_o * 2.0, ray_dir, step, stepi, dt, last_t * 2.0, mip, ray_t, marchi * 2);
            if res.hit {
                return res;
            }
        }
        let maski = vec3<i32>(t.xyz <= min(t.yzx, t.zxy));
        let mask = vec3<f32>(maski);
        last_t = min(t.x, min(t.y, t.z));
        t += mask * dt;
        marchi += maski * stepi;
    }
    
    return res;
}

fn mip2_loop(_o: vec3<f32>, ray_dir: vec3<f32>, step: vec3<f32>, stepi: vec3<i32>, dt: vec3<f32>, _last_t: f32, ray_t: f32, min_bound: vec3<i32>) -> MipResult {
    var res: MipResult;
    res.hit = false;
    res.color = vec4(0.0);

    #ifdef CHUNK_DEPTH_PREPASS
        if !is_resolution_enough(ray_t + _last_t * 4.0, 4.0) {
            res.hit = true;
            res.t = _last_t * 4.0;
            return res;
        }
    #endif

    var o = _o + ray_dir * _last_t;
    var marchi = vec3<i32>(floor(o) + 0.5);
    marchi = min(min_bound + 1, marchi);
    marchi = max(min_bound, marchi);
    var t = (step * (0.5 - (o - vec3<f32>(marchi))) + 0.5) * dt;
    var mod8 = marchi % 2;
    marchi -= mod8;

    var last_t = 0.0;
    for (var i = 0; i < 4; i += 1) {
        if (any(mod8 >= 2 || mod8 < 0)) {
            break;
        }
        var mip = textureLoad(voxels_mip1, marchi + mod8, 0).xy;
        if any(mip > 0u) {
            let res = mip1_loop(_o * 2.0, ray_dir, step, stepi, dt, (_last_t + last_t) * 2.0, mip, ray_t, (marchi + mod8) * 2);
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

fn mip1_loop(_o: vec3<f32>, ray_dir: vec3<f32>, step: vec3<f32>, stepi: vec3<i32>, dt: vec3<f32>, _last_t: f32, mip: vec2<u32>, ray_t: f32, min_bound: vec3<i32>) -> MipResult {
    var res: MipResult;
    res.hit = false;
    res.color = vec4(0.0);

    #ifdef CHUNK_DEPTH_PREPASS
        if !is_resolution_enough(ray_t + _last_t * 2.0, 2.0) {
            res.hit = true;
            res.t = _last_t * 2.0;
            return res;
        }
    #endif

    var o = _o + ray_dir * _last_t;
    var marchi = vec3<i32>(floor(o) + 0.5);
    marchi = min(min_bound + 1, marchi);
    marchi = max(min_bound, marchi);
    var t = (step * (0.5 - (o - vec3<f32>(marchi))) + 0.5) * dt;
    var mod4 = marchi % 2;
    marchi -= mod4;

    var last_t = 0.0;
    for (var i = 0; i < 4; i += 1) {
        if (any(mod4 >= 2 || mod4 < 0)) {
            break;
        }
        let m = vec3<u32>(mod4 > 0) << vec3(0u, 1u, 2u);
        let index = m.x | m.y | m.z;
        let comp = getMipByte(mip, index);
        if (comp > 0u) {
            res = mip0_loop(_o * 2.0, ray_dir, step, stepi, dt, (_last_t + last_t) * 2.0, comp, ray_t, (marchi + mod4) * 2);
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

fn mip0_loop(_o: vec3<f32>, ray_dir: vec3<f32>, step: vec3<f32>, stepi: vec3<i32>, dt: vec3<f32>, _last_t: f32, comp: u32, ray_t: f32, min_bound: vec3<i32>) -> MipResult {
    var res: MipResult;
    res.hit = false;
    res.color = vec4(0.0);

    #ifdef CHUNK_DEPTH_PREPASS
        if !is_resolution_enough(ray_t + _last_t, 1.0) {
            res.hit = true;
            res.t = _last_t;
            return res;
        }
    #endif

    var o = _o + ray_dir * _last_t;
    var marchi = vec3<i32>(o);
    marchi = min(min_bound + 1, marchi);
    marchi = max(min_bound, marchi);
    var t = (step * (0.5 - (o - vec3<f32>(marchi))) + 0.5) * dt;
    var mod2 = marchi % 2;
    marchi -= mod2;

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
                let voxel = textureLoad(voxels, marchi + mod2, 0).x;
                res.color = get_color(voxel);
                res.hit = true;
                res.t = _last_t + last_t;
                return res;
            #endif
        }
        let maski = vec3<i32>(t.xyz <= min(t.yzx, t.zxy));
        last_t = min(t.x, min(t.y, t.z));
        t += vec3<f32>(maski) * dt;
        mod2 += maski * stepi;
    }

    return res;
}
