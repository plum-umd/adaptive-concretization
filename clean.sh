#! /usr/bin/env bash
ls data/*.txt | grep -v stat_ | xargs rm -rf
rm -rf tmp/*.txt
rm -rf *.tmp
