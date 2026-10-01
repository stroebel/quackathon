shapefile_dir := "shapefiles"

# List available recipes
default:
    @just --list

# Install dependencies and unpack the shapefiles
setup: install shapefiles

# Install Python dependencies with uv
install:
    uv sync

# Unzip the MDB shapefiles into shapefiles/<name>/
shapefiles:
    #!/usr/bin/env bash
    set -euo pipefail
    for zip in MDBWards2026.zip MDBDistrictMunicipalities2026.zip; do
        name="${zip%.zip}"
        dest="{{shapefile_dir}}/$name"
        if [ -f "$dest/$name.shp" ]; then
            echo "$dest already present, skipping"
            continue
        fi
        mkdir -p "$dest"
        unzip -oq "$zip" -d "$dest"
        echo "Extracted $zip -> $dest"
    done

# Remove the extracted shapefiles
clean-shapefiles:
    rm -rf {{shapefile_dir}}
