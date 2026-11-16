package me.cortex.voxy.client.core.rendering.util;

import me.cortex.voxy.client.core.gpu.IGpuTexture;
import me.cortex.voxy.client.core.gpu.RenderBackendFactory;
import me.cortex.voxy.common.Logger;
import org.lwjgl.system.MemoryUtil;

import java.nio.ByteBuffer;

import static org.lwjgl.opengl.GL33C.*;

/**
 * Metal-path HiZ source: a max-depth pyramid built from MC's own depth
 * coverage of the current frame.
 *
 * <p>LODs only ever show where MC drew nothing (MC geometry is always
 * nearer than any LOD along a ray, since LODs live outside MC's convex
 * render square), so the GL path's HiZ effectively treats every MC-covered
 * pixel as "occludes everything". This class reproduces that cheaply:
 * <ol>
 *   <li>GL, end of frame N (MC opaque terrain drawn): reduce MC's depth to a
 *       quarter-resolution capture — 1.0 where any pixel is sky, else the
 *       farthest depth. Voxy itself renders at the SOLID head, BEFORE MC's
 *       terrain, so the current frame's depth is not available yet.</li>
 *   <li>GL, frame N+1: splat the eroded capture through the camera change
 *       into the current view ({@link #OCCLUDED} per covered texel) and read
 *       it back through a PBO (started early, mapped late).</li>
 *   <li>CPU: max-reduces the mip chain (power-of-two, ~0.2M texels).</li>
 *   <li>Metal: uploads each mip to a Shared R32F texture HOT samples.</li>
 * </ol>
 * Row 0 is the bottom row in both the GL readback and HOT's NDC-derived
 * texel coordinates, so no flip is needed. {@link #OCCLUDED} is positive
 * because screenspace.glsl treats 0.0 as "pyramid not built".
 *
 * <p>All raw GL state touched here is saved and restored so MC's
 * GlStateManager cache stays valid.
 */
public final class McCoverageHiz {
    private static final float OCCLUDED = 1.0e-6f;
    /** Pyramid base is the viewport's power-of-two size divided by 2^BASE_SHIFT. */
    private static final int BASE_SHIFT = 2;

    private static final String FULLSCREEN_VS = """
            #version 410 core
            void main() {
                vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
                gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
            }
            """;
    /** Capture: per low-res texel, 1.0 if any source pixel is sky, else the farthest MC depth. */
    private static final String REDUCE_FS = """
            #version 410 core
            uniform sampler2D depthTex;
            uniform ivec2 outSize;
            layout(location = 0) out float outDepth;
            void main() {
                ivec2 src = textureSize(depthTex, 0);
                ivec2 p = ivec2(gl_FragCoord.xy);
                ivec2 a = (p * src) / outSize;
                ivec2 b = min(((p + 1) * src + outSize - 1) / outSize, src);
                float m = 0.0;
                for (int y = a.y; y < b.y && m < 1.0; y++)
                    for (int x = a.x; x < b.x; x++)
                        m = max(m, texelFetch(depthTex, ivec2(x, y), 0).r);
                outDepth = m >= 1.0 ? 1.0 : m;
            }
            """;
    /**
     * Next frame: splat every fully-covered (eroded by one texel) capture
     * texel through the camera change into the current view. Untouched
     * texels stay "sky" — gaps only ever reduce culling.
     */
    private static final String SPLAT_VS = """
            #version 410 core
            uniform sampler2D prevDepth;
            uniform ivec2 size;
            uniform mat4 invPrevVP;
            uniform mat4 curVP;
            uniform vec3 camDelta;
            void main() {
                ivec2 t = ivec2(gl_VertexID % size.x, gl_VertexID / size.x);
                ivec2 hi = size - 1;
                float d = texelFetch(prevDepth, t, 0).r;
                float n = max(max(texelFetch(prevDepth, clamp(t + ivec2(1, 0), ivec2(0), hi), 0).r,
                                  texelFetch(prevDepth, clamp(t - ivec2(1, 0), ivec2(0), hi), 0).r),
                              max(texelFetch(prevDepth, clamp(t + ivec2(0, 1), ivec2(0), hi), 0).r,
                                  texelFetch(prevDepth, clamp(t - ivec2(0, 1), ivec2(0), hi), 0).r));
                if (max(d, n) >= 1.0) { gl_Position = vec4(-2.0, -2.0, -2.0, 1.0); return; }
                vec4 ndc = vec4((vec2(t) + 0.5) / vec2(size) * 2.0 - 1.0, d * 2.0 - 1.0, 1.0);
                vec4 p = invPrevVP * ndc;
                p /= p.w;
                gl_Position = curVP * vec4(p.xyz + camDelta, 1.0);
                // 2-texel points close the stretch gaps forward-splatting leaves
                // when the camera turns (a 1.0 gap max-propagates up every mip).
                gl_PointSize = 2.0;
            }
            """;
    private static final String SPLAT_FS = """
            #version 410 core
            layout(location = 0) out float coverage;
            void main() { coverage = %s; }
            """.formatted(Float.toString(OCCLUDED));

    private int reduceProgram, splatProgram, vao, prevFbo, prevTex, fbo, maskTex, pbo;
    private int uDepthTex, uOutSize, uPrevDepth, uSize, uInvPrevVP, uCurVP, uCamDelta;
    private final org.joml.Matrix4f invPrevVP = new org.joml.Matrix4f();
    private double prevCamX, prevCamY, prevCamZ;
    private boolean captured;
    private int baseW, baseH, levels;
    private long cpuPyramid;   // all mips, tightly packed floats
    private IGpuTexture texture;
    private boolean pending;
    private long fence;
    private long skipped;
    private static final long MAX_WAIT_NS = waitNs();
    private static long waitNs() {
        try { String v = me.cortex.voxy.client.core.util.VoxyTune.env("VOXY_HIZ_WAIT_US"); if (v != null) return Math.max(0, Long.parseLong(v.trim())) * 1000L; } catch (NumberFormatException ignored) {}
        return 1_000_000L; // late readbacks carry over a frame, so do not stall long
    }
    /** VOXY_HIZ_LATE=0: drop a late readback (old behaviour). Default: keep it
     *  in flight and consume it next frame (one frame staler, but HiZ stays on). */
    private static final boolean LATE_READBACK = !"0".equals(me.cortex.voxy.client.core.util.VoxyTune.env("VOXY_HIZ_LATE"));
    private boolean inFlight;
    /** Default: read the capture back right at the translucent head (GPU is
     *  long past it by the next frame, so the fence never stalls) and splat it
     *  on the CPU. VOXY_HIZ_CPU_SPLAT=0 restores the GL splat + late readback. */
    private static final boolean CPU_SPLAT = !"0".equals(me.cortex.voxy.client.core.util.VoxyTune.env("VOXY_HIZ_CPU_SPLAT"));
    private long cpuPrev;
    private float[] prevArr = new float[0], outArr = new float[0];
    private final float[] splatInv = new float[16], splatCur = new float[16];
    private float splatDx, splatDy, splatDz;
    // CPU path: capture ring (the GL queue can run >1 frame behind with shaders,
    // so a single PBO would be overwritten before it is ever ready).
    private static final int RING = 3;
    /** A held capture is re-splatted through the current camera for at most this many frames. */
    private static final int MAX_HELD_AGE = 4;
    private final int[] ringPbo = new int[RING];
    private final long[] ringFence = new long[RING];
    private final float[][] ringInv = new float[RING][16];
    private final double[][] ringCam = new double[RING][3];
    private int ringNext, useSlot = -1, heldAge = Integer.MAX_VALUE, reused;
    private final double[] heldCam = new double[3];
    private double curCamX, curCamY, curCamZ;
    private long late;
    public long skippedFrames() { return this.skipped; }
    private boolean broken;

    public IGpuTexture texture() { return this.texture; }
    public int packedSize() { return (this.baseW << 16) | this.baseH; }

    /**
     * End of frame N (after MC's opaque terrain, before translucents): reduce
     * the bound framebuffer's depth into the low-res capture texture and
     * remember the camera it was rendered with.
     */
    /** VOXY_HIZ_IRIS_DEPTH=0 disables the Iris depthtex0 fallback. */
    private static final boolean IRIS_DEPTH_FALLBACK = !"0".equals(System.getenv("VOXY_HIZ_IRIS_DEPTH"));

    public void capture(me.cortex.voxy.client.core.rendering.Viewport<?> vp) {
        this.captured = false;
        this.capCalls++;
        if (this.broken || vp.width <= 0 || vp.height <= 0) return;
        try {
            int bw = Math.max(1, Integer.highestOneBit(vp.width) >> BASE_SHIFT);
            int bh = Math.max(1, Integer.highestOneBit(vp.height) >> BASE_SHIFT);
            if (this.reduceProgram == 0) this.init();
            if (bw != this.baseW || bh != this.baseH) this.resize(bw, bh);

            int fb = glGetInteger(GL_DRAW_FRAMEBUFFER_BINDING);
            if (fb == 0) { this.capNoFb++; return; }
            int prevRead = glGetInteger(GL_READ_FRAMEBUFFER_BINDING);
            glBindFramebuffer(GL_READ_FRAMEBUFFER, fb);
            int type = glGetFramebufferAttachmentParameteri(GL_READ_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE);
            int depthTex = type == GL_TEXTURE
                    ? glGetFramebufferAttachmentParameteri(GL_READ_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME) : 0;
            glBindFramebuffer(GL_READ_FRAMEBUFFER, prevRead);
            // Iris: the bound terrain FBO may carry no texture depth attachment —
            // read the pipeline's depthtex0 directly (MC-only opaque depth at the
            // translucent head; LOD lives in its own vx targets under the contract).
            if (depthTex == 0 && IRIS_DEPTH_FALLBACK) depthTex = me.cortex.voxy.client.core.util.IrisUtil.irisMainDepthTexId();
            if (depthTex == 0) { this.capNoDepth++; return; }
            final int srcDepth = depthTex;
            this.runPass(this.prevFbo, false, () -> {
                glUseProgram(this.reduceProgram);
                glUniform1i(this.uDepthTex, 0);
                glUniform2i(this.uOutSize, this.baseW, this.baseH);
                glBindTexture(GL_TEXTURE_2D, srcDepth);
                glDrawArrays(GL_TRIANGLES, 0, 3);
                if (CPU_SPLAT) {
                    int slot = this.ringNext;
                    this.ringNext = (slot + 1) % RING;
                    if (this.ringFence[slot] != 0L) glDeleteSync(this.ringFence[slot]);
                    this.ringFence[slot] = this.readback(this.prevFbo, this.ringPbo[slot]);
                    new org.joml.Matrix4f(vp.vanillaProjection).mul(vp.modelView).invert().get(this.ringInv[slot]);
                    this.ringCam[slot][0] = vp.cameraX;
                    this.ringCam[slot][1] = vp.cameraY;
                    this.ringCam[slot][2] = vp.cameraZ;
                }
            });
            new org.joml.Matrix4f(vp.vanillaProjection).mul(vp.modelView).invert(this.invPrevVP);
            this.prevCamX = vp.cameraX;
            this.prevCamY = vp.cameraY;
            this.prevCamZ = vp.cameraZ;
            this.captured = true;
            this.capOk++;
        } catch (Throwable t) {
            Logger.error("McCoverageHiz disabled after GL failure", t);
            this.broken = true;
        }
    }

    /**
     * Frame N+1, before traversal: reproject the capture into the current
     * view and start the async readback. False (caller keeps the
     * always-pass pyramid) when there is no fresh capture.
     */
    /** Diagnostic counters (reset by {@link #diag()}). */
    private int capCalls, capNoFb, capNoDepth, capOk, beginCalls, beginNoCap;

    public String diag() {
        String s = "cap=" + this.capCalls + "/ok" + this.capOk + "/nofb" + this.capNoFb + "/nodepth" + this.capNoDepth
                + (this.broken ? "/BROKEN" : "") + " begin=" + this.beginCalls + "/nocap" + this.beginNoCap + " late=" + this.late + " reuse=" + this.reused;
        this.capCalls = this.capNoFb = this.capNoDepth = this.capOk = this.beginCalls = this.beginNoCap = 0;
        this.late = 0;
        this.reused = 0;
        return s;
    }

    public boolean begin(me.cortex.voxy.client.core.rendering.Viewport<?> vp) {
        this.pending = false;
        this.beginCalls++;
        if (!this.captured) this.beginNoCap++;
        if (this.broken) return false;
        if (this.inFlight && this.fence != 0L) {
            // Previous readback timed out last frame: consume it instead of
            // issuing a new one (the GPU is busy anyway).
            this.inFlight = false;
            this.captured = false;
            this.pending = true;
            this.late++;
            return true;
        }
        if (CPU_SPLAT) {
            this.captured = false;
            int bw = Math.max(1, Integer.highestOneBit(vp.width) >> BASE_SHIFT);
            int bh = Math.max(1, Integer.highestOneBit(vp.height) >> BASE_SHIFT);
            if (bw != this.baseW || bh != this.baseH || this.reduceProgram == 0) return false;
            // Newest ready capture wins; older ones are obsolete.
            this.useSlot = -1;
            for (int k = 1; k <= RING; k++) {
                int slot = Math.floorMod(this.ringNext - k, RING);
                long f = this.ringFence[slot];
                if (f == 0L) continue;
                if (this.useSlot >= 0) { glDeleteSync(f); this.ringFence[slot] = 0L; continue; }
                if (glGetSynci(f, GL_SYNC_STATUS, null) == GL_SIGNALED) this.useSlot = slot;
            }
            if (this.useSlot < 0) {
                // Nothing new yet: reproject the held capture again (bounded age).
                if (this.heldAge >= MAX_HELD_AGE) { this.skipped++; return false; }
                this.reused++;
            }
            new org.joml.Matrix4f(vp.vanillaProjection).mul(vp.modelView).get(this.splatCur);
            this.curCamX = vp.cameraX;
            this.curCamY = vp.cameraY;
            this.curCamZ = vp.cameraZ;
            this.pending = true;
            return true;
        }
        if (!this.captured) return false;
        this.captured = false;
        int bw = Math.max(1, Integer.highestOneBit(vp.width) >> BASE_SHIFT);
        int bh = Math.max(1, Integer.highestOneBit(vp.height) >> BASE_SHIFT);
        if (bw != this.baseW || bh != this.baseH) return false;
        try {
            var curVP = new org.joml.Matrix4f(vp.vanillaProjection).mul(vp.modelView);
            float[] inv = new float[16], cur = new float[16];
            this.invPrevVP.get(inv);
            curVP.get(cur);
            float dx = (float) (this.prevCamX - vp.cameraX);
            float dy = (float) (this.prevCamY - vp.cameraY);
            float dz = (float) (this.prevCamZ - vp.cameraZ);
            this.runPass(this.fbo, true, () -> {
                glUseProgram(this.splatProgram);
                glUniform1i(this.uPrevDepth, 0);
                glUniform2i(this.uSize, this.baseW, this.baseH);
                glUniformMatrix4fv(this.uInvPrevVP, false, inv);
                glUniformMatrix4fv(this.uCurVP, false, cur);
                glUniform3f(this.uCamDelta, dx, dy, dz);
                glBindTexture(GL_TEXTURE_2D, this.prevTex);
                boolean progPointSize = org.lwjgl.opengl.GL11C.glIsEnabled(org.lwjgl.opengl.GL32C.GL_PROGRAM_POINT_SIZE);
                if (!progPointSize) org.lwjgl.opengl.GL11C.glEnable(org.lwjgl.opengl.GL32C.GL_PROGRAM_POINT_SIZE);
                glDrawArrays(GL_POINTS, 0, this.baseW * this.baseH);
                if (!progPointSize) org.lwjgl.opengl.GL11C.glDisable(org.lwjgl.opengl.GL32C.GL_PROGRAM_POINT_SIZE);

                if (this.fence != 0L) glDeleteSync(this.fence);
                this.fence = this.readback(this.fbo, this.pbo);
            });
            this.pending = true;
            return true;
        } catch (Throwable t) {
            Logger.error("McCoverageHiz disabled after GL failure", t);
            this.broken = true;
            return false;
        }
    }

    /** Waits for the readback started by {@link #begin}, builds the mips, uploads to Metal. */
    public boolean finish() {
        if (!this.pending) return false;
        this.pending = false;
        if (CPU_SPLAT) {
            if (this.useSlot >= 0) {
                int slot = this.useSlot;
                int prevPbo = glGetInteger(GL_PIXEL_PACK_BUFFER_BINDING);
                glBindBuffer(GL_PIXEL_PACK_BUFFER, this.ringPbo[slot]);
                ByteBuffer mapped = glMapBufferRange(GL_PIXEL_PACK_BUFFER, 0, (long) this.baseW * this.baseH * 4, GL_MAP_READ_BIT);
                if (mapped != null) {
                    MemoryUtil.memCopy(MemoryUtil.memAddress(mapped), this.cpuPrev, (long) this.baseW * this.baseH * 4);
                    glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
                }
                glBindBuffer(GL_PIXEL_PACK_BUFFER, prevPbo);
                glDeleteSync(this.ringFence[slot]);
                this.ringFence[slot] = 0L;
                if (mapped == null) { this.heldAge = Integer.MAX_VALUE; this.skipped++; return false; }
                System.arraycopy(this.ringInv[slot], 0, this.splatInv, 0, 16);
                System.arraycopy(this.ringCam[slot], 0, this.heldCam, 0, 3);
                this.heldAge = 0;
            } else {
                this.heldAge++;
            }
            this.splatDx = (float) (this.heldCam[0] - this.curCamX);
            this.splatDy = (float) (this.heldCam[1] - this.curCamY);
            this.splatDz = (float) (this.heldCam[2] - this.curCamZ);
            this.cpuSplat();
            this.buildMips();
            return true;
        }
        if (this.fence != 0L) {
            // Bounded wait: a slow GL frame skips HiZ for one frame (= no culling,
            // always correct) instead of stalling the render thread on the map.
            int st = glClientWaitSync(this.fence, GL_SYNC_FLUSH_COMMANDS_BIT, MAX_WAIT_NS);
            if (st == GL_TIMEOUT_EXPIRED && LATE_READBACK && !CPU_SPLAT) { this.skipped++; this.inFlight = true; return false; }
            glDeleteSync(this.fence);
            this.fence = 0L;
            if (st == GL_TIMEOUT_EXPIRED || st == GL_WAIT_FAILED) { this.skipped++; return false; }
        }
        int prevPbo = glGetInteger(GL_PIXEL_PACK_BUFFER_BINDING);
        glBindBuffer(GL_PIXEL_PACK_BUFFER, this.pbo);
        ByteBuffer mapped = glMapBufferRange(GL_PIXEL_PACK_BUFFER, 0, (long) this.baseW * this.baseH * 4, GL_MAP_READ_BIT);
        if (mapped == null) {
            glBindBuffer(GL_PIXEL_PACK_BUFFER, prevPbo);
            return false;
        }
        MemoryUtil.memCopy(MemoryUtil.memAddress(mapped), CPU_SPLAT ? this.cpuPrev : this.cpuPyramid, (long) this.baseW * this.baseH * 4);
        glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
        glBindBuffer(GL_PIXEL_PACK_BUFFER, prevPbo);
        this.buildMips();
        return true;
    }

    private void buildMips() {
        // Max-reduce: each power-of-two level halves cleanly (clamped at 1).
        long src = this.cpuPyramid;
        int w = this.baseW, h = this.baseH;
        this.texture.uploadSubImage2D(0, 0, 0, w, h, GL_RED, GL_FLOAT, src);
        for (int level = 1; level < this.levels; level++) {
            int nw = Math.max(1, w >> 1), nh = Math.max(1, h >> 1);
            long dst = src + (long) w * h * 4;
            for (int y = 0; y < nh; y++) {
                int y0 = Math.min(y * 2, h - 1), y1 = Math.min(y * 2 + 1, h - 1);
                for (int x = 0; x < nw; x++) {
                    int x0 = Math.min(x * 2, w - 1), x1 = Math.min(x * 2 + 1, w - 1);
                    float m = Math.max(
                            Math.max(MemoryUtil.memGetFloat(src + ((long) y0 * w + x0) * 4),
                                     MemoryUtil.memGetFloat(src + ((long) y0 * w + x1) * 4)),
                            Math.max(MemoryUtil.memGetFloat(src + ((long) y1 * w + x0) * 4),
                                     MemoryUtil.memGetFloat(src + ((long) y1 * w + x1) * 4)));
                    MemoryUtil.memPutFloat(dst + ((long) y * nw + x) * 4, m);
                }
            }
            this.texture.uploadSubImage2D(level, 0, 0, nw, nh, GL_RED, GL_FLOAT, dst);
            src = dst; w = nw; h = nh;
        }
    }

    /** Reads {@code fb}'s colour 0 into the PBO and fences it (pack state saved). */
    private long readback(int fb, int pboId) {
        glBindFramebuffer(GL_READ_FRAMEBUFFER, fb);
        glReadBuffer(GL_COLOR_ATTACHMENT0);
        int prevPackRow = glGetInteger(GL_PACK_ROW_LENGTH);
        int prevPackSkipRows = glGetInteger(GL_PACK_SKIP_ROWS);
        int prevPackSkipPixels = glGetInteger(GL_PACK_SKIP_PIXELS);
        int prevPackAlign = glGetInteger(GL_PACK_ALIGNMENT);
        glPixelStorei(GL_PACK_ROW_LENGTH, 0);
        glPixelStorei(GL_PACK_SKIP_ROWS, 0);
        glPixelStorei(GL_PACK_SKIP_PIXELS, 0);
        glPixelStorei(GL_PACK_ALIGNMENT, 4);
        glBindBuffer(GL_PIXEL_PACK_BUFFER, pboId);
        glReadPixels(0, 0, this.baseW, this.baseH, GL_RED, GL_FLOAT, 0L);
        long f = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
        glFlush();
        glPixelStorei(GL_PACK_ROW_LENGTH, prevPackRow);
        glPixelStorei(GL_PACK_SKIP_ROWS, prevPackSkipRows);
        glPixelStorei(GL_PACK_SKIP_PIXELS, prevPackSkipPixels);
        glPixelStorei(GL_PACK_ALIGNMENT, prevPackAlign);
        return f;
    }

    /**
     * CPU twin of SPLAT_VS/SPLAT_FS: forward-splats every eroded covered texel
     * of {@link #cpuPrev} through the camera change as a 2x2-texel point into
     * pyramid level 0 (1.0 = sky). Same clip and point rasterisation rules as GL.
     */
    private void cpuSplat() {
        final int W = this.baseW, H = this.baseH, n = W * H;
        if (this.prevArr.length != n) { this.prevArr = new float[n]; this.outArr = new float[n]; }
        final float[] p = this.prevArr, o = this.outArr, inv = this.splatInv, cur = this.splatCur;
        MemoryUtil.memFloatBuffer(this.cpuPrev, n).get(p);
        java.util.Arrays.fill(o, 1.0f);
        final float dx = this.splatDx, dy = this.splatDy, dz = this.splatDz;
        for (int y = 0; y < H; y++) {
            int ym = Math.max(y - 1, 0), yp = Math.min(y + 1, H - 1);
            float ny = (y + 0.5f) / H * 2f - 1f;
            for (int x = 0; x < W; x++) {
                int i = y * W + x;
                float d = p[i];
                if (d >= 1f) continue;
                float m = Math.max(Math.max(p[y * W + Math.min(x + 1, W - 1)], p[y * W + Math.max(x - 1, 0)]),
                                   Math.max(p[yp * W + x], p[ym * W + x]));
                if (m >= 1f) continue;
                float nx = (x + 0.5f) / W * 2f - 1f, nz = d * 2f - 1f;
                float ww = inv[3] * nx + inv[7] * ny + inv[11] * nz + inv[15];
                float px = (inv[0] * nx + inv[4] * ny + inv[8] * nz + inv[12]) / ww + dx;
                float py = (inv[1] * nx + inv[5] * ny + inv[9] * nz + inv[13]) / ww + dy;
                float pz = (inv[2] * nx + inv[6] * ny + inv[10] * nz + inv[14]) / ww + dz;
                float cw = cur[3] * px + cur[7] * py + cur[11] * pz + cur[15];
                if (!(cw > 0f)) continue;
                float cx = cur[0] * px + cur[4] * py + cur[8] * pz + cur[12];
                float cy = cur[1] * px + cur[5] * py + cur[9] * pz + cur[13];
                float cz = cur[2] * px + cur[6] * py + cur[10] * pz + cur[14];
                if (Math.abs(cx) > cw || Math.abs(cy) > cw || Math.abs(cz) > cw) continue;
                float sx = (cx / cw * 0.5f + 0.5f) * W, sy = (cy / cw * 0.5f + 0.5f) * H;
                // gl_PointSize 2: texels whose centres fall in [s-1, s+1)
                int x0 = Math.max((int) Math.ceil(sx - 1.5f), 0), x1 = Math.min((int) Math.ceil(sx + 0.5f) - 1, W - 1);
                int y0 = Math.max((int) Math.ceil(sy - 1.5f), 0), y1 = Math.min((int) Math.ceil(sy + 0.5f) - 1, H - 1);
                for (int ty = y0; ty <= y1; ty++)
                    for (int tx = x0; tx <= x1; tx++) o[ty * W + tx] = OCCLUDED;
            }
        }
        MemoryUtil.memFloatBuffer(this.cpuPyramid, n).put(o);
    }

    private void init() {
        this.reduceProgram = link(FULLSCREEN_VS, REDUCE_FS);
        this.splatProgram = link(SPLAT_VS, SPLAT_FS);
        this.uDepthTex = glGetUniformLocation(this.reduceProgram, "depthTex");
        this.uOutSize = glGetUniformLocation(this.reduceProgram, "outSize");
        this.uPrevDepth = glGetUniformLocation(this.splatProgram, "prevDepth");
        this.uSize = glGetUniformLocation(this.splatProgram, "size");
        this.uInvPrevVP = glGetUniformLocation(this.splatProgram, "invPrevVP");
        this.uCurVP = glGetUniformLocation(this.splatProgram, "curVP");
        this.uCamDelta = glGetUniformLocation(this.splatProgram, "camDelta");
        this.vao = glGenVertexArrays();
        this.prevFbo = glGenFramebuffers();
        this.prevTex = glGenTextures();
        this.fbo = glGenFramebuffers();
        this.maskTex = glGenTextures();
        this.pbo = glGenBuffers();
        for (int i = 0; i < RING; i++) this.ringPbo[i] = glGenBuffers();
    }

    private static int link(String vsSrc, String fsSrc) {
        int vs = compile(GL_VERTEX_SHADER, vsSrc);
        int fs = compile(GL_FRAGMENT_SHADER, fsSrc);
        int program = glCreateProgram();
        glAttachShader(program, vs);
        glAttachShader(program, fs);
        glLinkProgram(program);
        glDeleteShader(vs);
        glDeleteShader(fs);
        if (glGetProgrami(program, GL_LINK_STATUS) == 0) {
            throw new IllegalStateException("McCoverageHiz link: " + glGetProgramInfoLog(program));
        }
        return program;
    }

    private static int compile(int stage, String src) {
        int s = glCreateShader(stage);
        glShaderSource(s, src);
        glCompileShader(s);
        if (glGetShaderi(s, GL_COMPILE_STATUS) == 0) {
            throw new IllegalStateException("McCoverageHiz compile: " + glGetShaderInfoLog(s));
        }
        return s;
    }

    private void resize(int bw, int bh) {
        this.baseW = bw;
        this.baseH = bh;
        this.levels = 32 - Integer.numberOfLeadingZeros(Math.max(bw, bh));

        int prevActive = glGetInteger(GL_ACTIVE_TEXTURE);
        glActiveTexture(GL_TEXTURE0);
        int prevBound = glGetInteger(GL_TEXTURE_BINDING_2D);
        int prevDraw = glGetInteger(GL_DRAW_FRAMEBUFFER_BINDING);
        for (int[] pair : new int[][]{{this.maskTex, this.fbo}, {this.prevTex, this.prevFbo}}) {
            glBindTexture(GL_TEXTURE_2D, pair[0]);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, bw, bh, 0, GL_RED, GL_FLOAT, 0L);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, pair[1]);
            glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, pair[0], 0);
        }
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, prevDraw);
        glBindTexture(GL_TEXTURE_2D, prevBound);
        glActiveTexture(prevActive);
        this.captured = false;

        int prevPbo = glGetInteger(GL_PIXEL_PACK_BUFFER_BINDING);
        glBindBuffer(GL_PIXEL_PACK_BUFFER, this.pbo);
        glBufferData(GL_PIXEL_PACK_BUFFER, (long) bw * bh * 4, GL_STREAM_READ);
        for (int i = 0; i < RING; i++) {
            if (this.ringFence[i] != 0L) { glDeleteSync(this.ringFence[i]); this.ringFence[i] = 0L; }
            glBindBuffer(GL_PIXEL_PACK_BUFFER, this.ringPbo[i]);
            glBufferData(GL_PIXEL_PACK_BUFFER, (long) bw * bh * 4, GL_STREAM_READ);
        }
        this.heldAge = Integer.MAX_VALUE;
        glBindBuffer(GL_PIXEL_PACK_BUFFER, prevPbo);

        long total = 0;
        for (int l = 0, w = bw, h = bh; l < this.levels; l++, w = Math.max(1, w >> 1), h = Math.max(1, h >> 1)) {
            total += (long) w * h * 4;
        }
        if (this.cpuPyramid != 0) MemoryUtil.nmemFree(this.cpuPyramid);
        if (this.cpuPrev != 0) MemoryUtil.nmemFree(this.cpuPrev);
        this.cpuPrev = MemoryUtil.nmemAlloc((long) bw * bh * 4);
        this.cpuPyramid = MemoryUtil.nmemAlloc(total);

        if (this.texture != null) this.texture.free();
        this.texture = RenderBackendFactory.get().createTexture()
                .storeUploadable(GL_R32F, this.levels, bw, bh)
                .name("HiZ-MCCoverage");
    }

    /** Runs {@code body} into {@code target} with every raw GL state it touches saved and restored. */
    private void runPass(int target, boolean clearToSky, Runnable body) {
        int prevProgram = glGetInteger(GL_CURRENT_PROGRAM);
        int prevVao = glGetInteger(GL_VERTEX_ARRAY_BINDING);
        int prevDraw = glGetInteger(GL_DRAW_FRAMEBUFFER_BINDING);
        int prevRead = glGetInteger(GL_READ_FRAMEBUFFER_BINDING);
        int prevActive = glGetInteger(GL_ACTIVE_TEXTURE);
        glActiveTexture(GL_TEXTURE0);
        int prevTex0 = glGetInteger(GL_TEXTURE_BINDING_2D);
        int prevSampler0 = glGetInteger(GL_SAMPLER_BINDING);
        int prevPbo = glGetInteger(GL_PIXEL_PACK_BUFFER_BINDING);
        int[] prevViewport = new int[4];
        glGetIntegerv(GL_VIEWPORT, prevViewport);
        boolean depthTest = glIsEnabled(GL_DEPTH_TEST);
        boolean stencilTest = glIsEnabled(GL_STENCIL_TEST);
        boolean blend = glIsEnabled(GL_BLEND);
        boolean cull = glIsEnabled(GL_CULL_FACE);
        boolean scissor = glIsEnabled(GL_SCISSOR_TEST);
        boolean[] colorMask = new boolean[4];
        try (var stack = org.lwjgl.system.MemoryStack.stackPush()) {
            var cm = stack.malloc(4);
            glGetBooleanv(GL_COLOR_WRITEMASK, cm);
            for (int i = 0; i < 4; i++) colorMask[i] = cm.get(i) != 0;
        }

        glDisable(GL_DEPTH_TEST);
        glDisable(GL_STENCIL_TEST);
        glDisable(GL_BLEND);
        glDisable(GL_CULL_FACE);
        glDisable(GL_SCISSOR_TEST);
        glColorMask(true, true, true, true);

        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, target);
        glViewport(0, 0, this.baseW, this.baseH);
        if (clearToSky) glClearBufferfv(GL_COLOR, 0, new float[]{1.0f, 0, 0, 0});
        glBindSampler(0, 0);
        glBindVertexArray(this.vao);
        try {
            body.run();
        } finally {
            glBindBuffer(GL_PIXEL_PACK_BUFFER, prevPbo);
            glBindVertexArray(prevVao);
            glBindSampler(0, prevSampler0);
            glBindTexture(GL_TEXTURE_2D, prevTex0);
            glActiveTexture(prevActive);
            glUseProgram(prevProgram);
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, prevDraw);
            glBindFramebuffer(GL_READ_FRAMEBUFFER, prevRead);
            glViewport(prevViewport[0], prevViewport[1], prevViewport[2], prevViewport[3]);
            glColorMask(colorMask[0], colorMask[1], colorMask[2], colorMask[3]);
            if (depthTest) glEnable(GL_DEPTH_TEST);
            if (stencilTest) glEnable(GL_STENCIL_TEST);
            if (blend) glEnable(GL_BLEND);
            if (cull) glEnable(GL_CULL_FACE);
            if (scissor) glEnable(GL_SCISSOR_TEST);
        }
    }

    public void free() {
        if (this.fence != 0L) { glDeleteSync(this.fence); this.fence = 0L; }
        for (int i = 0; i < RING; i++) if (this.ringFence[i] != 0L) { glDeleteSync(this.ringFence[i]); this.ringFence[i] = 0L; }
        if (this.reduceProgram != 0) {
            glDeleteProgram(this.reduceProgram);
            glDeleteProgram(this.splatProgram);
            glDeleteVertexArrays(this.vao);
            glDeleteFramebuffers(this.fbo);
            glDeleteFramebuffers(this.prevFbo);
            glDeleteTextures(this.maskTex);
            glDeleteTextures(this.prevTex);
            glDeleteBuffers(this.pbo);
            for (int i = 0; i < RING; i++) glDeleteBuffers(this.ringPbo[i]);
            this.reduceProgram = 0;
        }
        if (this.cpuPyramid != 0) {
            MemoryUtil.nmemFree(this.cpuPyramid);
            this.cpuPyramid = 0;
        }
        if (this.cpuPrev != 0) {
            MemoryUtil.nmemFree(this.cpuPrev);
            this.cpuPrev = 0;
        }
        if (this.texture != null) {
            this.texture.free();
            this.texture = null;
        }
    }
}
