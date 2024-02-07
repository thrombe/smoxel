{
  description = "yaaaaaaaaaaaaaaaaaaaaa";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-23.11";
    nixpkgs-unstable.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = inputs@{self, ...}:
    inputs.flake-utils.lib.eachSystem ["x86_64-linux"] (system: let
      pkgs = import inputs.nixpkgs {
        inherit system;
      };
      unstable = import inputs.nixpkgs-unstable {
        inherit system;
      };

      manifest = (pkgs.lib.importTOML ./Cargo.toml).package;

      # - [bevy/docs/linux_dependencies.md at main · bevyengine/bevy · GitHub](https://github.com/bevyengine/bevy/blob/main/docs/linux_dependencies.md#nixos)
      packages = pkgs: with pkgs; [
          # - [RPATH, or why lld doesn't work on NixOS](https://matklad.github.io/2022/03/14/rpath-or-why-lld-doesnt-work-on-nixos.html)
          llvmPackages.bintools
          llvmPackages_15.clang
          # - [Using mold as linker prevents - NixOS Discourse](https://discourse.nixos.org/t/using-mold-as-linker-prevents-libraries-from-being-found/18530/5)
          # mold
          unstable.mold-wrapped

          openssl
          xdotool
          alsa-oss
          alsa-lib
          systemd

          libGL

          udev
          alsa-lib
          vulkan-loader
          vulkan-headers
          vulkan-validation-layers
          vulkan-tools
          glslang

          libxkbcommon
          wayland

          xorg.libXcursor
          xorg.libXrandr
          xorg.libXi
          xorg.libX11

          pkg-config
        ];
    in {
      packages.default = unstable.rustPlatform.buildRustPackage {
        pname = manifest.name;
        version = manifest.version;
        cargoLock.lockFile = ./Cargo.lock;
        # src = pkgs.lib.cleanSource ./.;

        # - [nix flake rust and pkgconfig](https://discourse.nixos.org/t/nix-and-rust-how-to-use-pkgconfig/17465/3)
        buildInputs = packages pkgs;
        nativeBuildInputs = with pkgs; [
          pkg-config
        ];
      };

      devShells.default = pkgs.mkShell {
        nativeBuildInputs = with pkgs;
          [
            unstable.rust-analyzer
            unstable.rustfmt
            unstable.clippy
            renderdoc
            (pkgs.buildFHSEnv {
              name = "game";
              targetPkgs = packages;
              runScript = ''
                #!/usr/bin/env bash
                nvidia-offload ./target/debug/${manifest.name}
              '';
            })
            (pkgs.buildFHSEnv {
              name = "game-release";
              targetPkgs = packages;
              runScript = ''
                #!/usr/bin/env bash
                nvidia-offload ./target/release/${manifest.name}
              '';
            })
            (pkgs.buildFHSEnv {
              name = "renderdoc";
              targetPkgs = packages;
              runScript = ''
                #!/usr/bin/env bash
                nvidia-offload ${renderdoc}/bin/qrenderdoc
              '';
            })
            (pkgs.buildFHSEnv {
              name = "fhs-shell";
              targetPkgs = packages;
            })
          ]
          ++ self.packages."${system}".default.nativeBuildInputs
          ++ self.packages."${system}".default.buildInputs;

        LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath self.packages."${system}".default.buildInputs;

        shellHook = ''
          export RUST_BACKTRACE="1"
          # export RUSTFLAGS="-C target-feature=-crt-static"
        '';
      };
    });
}
