#!/usr/bin/env bash

if [ -e "$1" ]
	then echo "Fixing dateline for $1"
	else
		echo "File not found: $1"
		exit 1
fi

# the correct bounds for the map
L="20037508.342789244"

# figure out what is the full extent of the x variable
Lmax=$( gdalinfo $1 | sed -n 's/Upper Right (\([0-9]*.[0-9]*\),.*/\1/p' )
Ldiff=$( echo "scale=5; $Lmax - $L" | bc )
Lbnd=$( echo "scale=5; -$L + $Ldiff" | bc )

if (( $(echo "$Ldiff < 0" | bc -l) )); then
	echo "No need to wrap dateline"
	exit 0
fi

echo "Need to fix dateline"

# export two intermediate files
fname=$( basename $1 .tiff )

# first the main part
gdal_translate -projwin -$L $L $L -$L $1 "$fname"_main.tiff &&

# then the extra part (past dateline)
gdal_translate -projwin $L $L $Lmax -$L -a_ullr -$L $L $Lbnd -$L $1 "$fname"_extra.tiff &&

# merge
gdal_merge.py -ul_lr -$L $L $L -$L -n 0 -of GTiff -o "$fname"_centered.tiff "$fname"_main.tiff "$fname"_extra.tiff &&

rm "$fname"_main.tiff "$fname"_extra.tiff



