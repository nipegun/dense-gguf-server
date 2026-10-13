# DGS licenses and notices

Dense GGUF Server is a project developed and maintained by NiPeGun. The project's
`LICENSE` file contains the PolyForm Noncommercial License 1.0.0 for NiPeGun's
contributions. Third-party portions retain their own licenses and copyright
notices; the project license does not replace them.

| Component | Notice |
| --- | --- |
| llama.cpp / ggml code | `LICENSE-llama.cpp`, `AUTHORS` |
| CPU matrix kernels, Mozilla Foundation | `LICENSE-cpu-kernels` |
| cpp-httplib | `LICENSE-cpp-httplib` |
| nlohmann/json | `LICENSE-jsonhpp` |
| Swagger UI | `LICENSE-swagger-ui`, `NOTICE-swagger-ui` |
| stb_image | `LICENSE-stb-image` |
| miniaudio | `LICENSE-miniaudio` |
| subprocess.h | `LICENSE-subprocess` |
| Base64 helper | `LICENSE-base64` |
| Hash helpers | `LICENSE-xxhash`, `LICENSE-rotate-bits`, `LICENSE-sha256`, `LICENSE-sha1` |
| Vendored web mathematics libraries | `LICENSE-big-integer`, `LICENSE-decimal-js`, `LICENSE-nerdamer` |

The copies here supplement the notices kept in dependency sources and bundled
web assets. Preserve those notices when redistributing or updating dependencies.
Model weights and external system libraries have their own terms.

`cmake --install _/build --prefix PREFIX` installs the project's `LICENSE` and this
directory at `PREFIX/share/doc/dgs-server/licenses/` with the default data layout.
The `Licenses` CMake component can also be installed on its own. When distributing
the executable manually, include the project license and these notices alongside it.
