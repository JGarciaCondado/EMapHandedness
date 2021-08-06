#!/bin/bash

# Script to download files from EMDB http file download services.
# Use the -h switch to get help on usage.

if ! command -v rsync &> /dev/null
then
    echo "'rsync' could not be found. You need to install 'rsync' for this script to work."
    exit 1
fi

PROGNAME=$0
BASE_URL="rsync.rcsb.org::emdb/structures"

usage() {
  cat << EOF >&2
Usage: $PROGNAME -f <file> [-o <dir>] [-c] [-p]

 -f <file>: the input file containing a comma-separated list of EMDB ids
 -o  <dir>: the output dir, default: current dir
EOF
  exit 1
}

download() {
  url="$BASE_URL/$1"
  out=$2
  echo "Downloading $url to $out"
  rsync -rlpt -v -z --delete --port=33444 $url $out
  echo ""
}

listfile=""
outdir="."
while getopts f:o:cpaxsmr o
do
  case $o in
    (f) listfile=$OPTARG;;
    (o) outdir=$OPTARG;;
    (*) usage
  esac
done
shift "$((OPTIND - 1))"

if [ "$listfile" == "" ]
then
  echo "Parameter -f must be provided"
  exit 1
fi
contents=$(cat $listfile)

# see https://stackoverflow.com/questions/918886/how-do-i-split-a-string-on-a-delimiter-in-bash#tab-top
IFS=',' read -ra tokens <<< "$contents"

for token in "${tokens[@]}"
do
    download ${token} $outdir
done








