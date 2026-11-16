package me.cortex.voxy.client.core.util;

import java.io.IOException;
import java.nio.file.Files;
import java.util.Properties;

/**
 * Env-var lookup with a file fallback, so Metal perf switches can be A/B'd
 * without editing the launcher's environment: a key in
 * config/voxy-tune.properties (e.g. {@code VOXY_LOD_NO_DISCARD=1}) acts like
 * the env var. Read fresh on each call — callers are pipeline construction
 * only, so a resource reload (F3+T) picks up edits.
 */
public final class VoxyTune {
    private VoxyTune() {}

    public static String env(String name) {
        String v = System.getenv(name);
        if (v != null) return v;
        try {
            var f = net.fabricmc.loader.api.FabricLoader.getInstance().getConfigDir().resolve("voxy-tune.properties");
            if (!Files.isRegularFile(f)) return null;
            var p = new Properties();
            try (var in = Files.newInputStream(f)) {
                p.load(in);
            }
            String s = p.getProperty(name);
            return s == null ? null : s.trim();
        } catch (IOException | RuntimeException e) {
            return null;
        }
    }
}
