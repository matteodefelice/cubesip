# GOAL
Creating a Python software to retrieve remote sensing and climate data from different sources for regional or site-specific applications. The name of this software is "cubesip".
The target user is an analyst or researcher who needs a quick and smart way to retrieve the data needed to calculate specific metrics on nature/climate/sustainability domain

# HIGH-LEVEL WORKFLOW
The user requests (calling a Python function for now):
- a specific dataset on one of the available drivers (openEO, CDSE openEO Federation, arbitrary STAC, EarthDataHub, TESSERA embeddings or local filesystem)
- the set of bands / variables that are needed
- an area using geohash (for example `qem7`)
- a time slice between two days (it can also be NULL if the dataset does not have a temporal dimension)
- target directory (for now only local filesystem)
- addictional input parameters (optional)

After the request, the code checks if the data is already available (checking the manifest file) or if the data exist on the external source (according to the driver), the size of the request. 

At this point the user can decide to run the request and wait for the results. The results consist of:
1. The data in a NetCDF format
2. The item to add to the manifest file

# COMPONENTS

## YAML
A YAML file should contain the login credentials/tokens to access openEO and Earth Data Hub. Also, it should contain the path to the manifest file. 
## MANIFEST
The MANIFEST contains a table with the information contained in the request and:
1. the timestamp of the request
2. the user / machine name who run the request
3. the size of the request
4. the path (URI) to the NetCDF file

It makes sense when multiple bands/variables are contained in the same file to "spread" the file on multiple rows in the manifest file, a row for each variable. 

## REQUEST
The request process is already described in the high-level workflow

## CHECKER
The code checking the request should check if:
1. the dataset requested exists on the server 
2. if in the manifest file there is the data requested considering also geohash hierarchy, in other words you can reuse existing data to extract a subset (temporal/spatial)

## DRIVERS

Five drivers, each with one meaning for `dataset`:

- `openeo`: collections (for example `SENTINEL2_L2A`) of one openEO back-end, any of those listed on https://hub.openeo.org; the default is the Copernicus Data Space Ecosystem (https://openeo.dataspace.copernicus.eu)
- `cdse_federation`: collections of the Copernicus Data Space Ecosystem openEO Federation (https://openeofed.dataspace.copernicus.eu), which federates CDSE and Terrascope (for example `TERRASCOPE_S2_NDVI_V2`, `AGERA5`)
- `stac`: arbitrary STAC collections (for example https://planetarycomputer.microsoft.com/api/stac/v1/collections/landsat-c2-l2), loaded using `load_stac` on an openEO back-end (`openeo` or `cdse_federation`)

Earth Data Hub can be used to access ERA5 or other datasets in the catalogue https://earthdatahub.destine.eu/catalogue

The `tessera` driver gives access to the TESSERA foundation-model embeddings (https://github.com/ucam-eo/tessera), annual 128-channel embeddings at 10 m, read with the Zarr API of geotessera (https://github.com/ucam-eo/geotessera)

The `local` driver can be used to access a Zarr store existent in the given path or any other spatial format. 


# REQUIREMENTS
- The request should be provided with a JSON 
- The config YAML should be put in the gitignore
- Check the documentation and use the right API
- Use xarray and zarr to access Earth Data Hub (EDH) data

# FUTURE WORK
Not in the scope of cubesip:
- Manifest file stored on a database
- Batch jobs for openEO
- Saving file on cloud storage (azure, s3)

# USE CASES
An analyst want to retrieve the NDVI for a specific year for a specific farm. She finds the geohash containing the farm location and she finds the right dataset/collection. If the data contains the multiple bands, she will calculate the NDVI by herself. 
