export ICU_PREFIX=$(brew --prefix icu4c@78)
export CFLAGS="-I$ICU_PREFIX/include"
export LDFLAGS="-L$ICU_PREFIX/lib -Wl,-rpath,$ICU_PREFIX/lib"
export PKG_CONFIG_PATH="$ICU_PREFIX/lib/pkgconfig"
uv add --no-binary pyicu