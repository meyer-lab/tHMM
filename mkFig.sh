#!/bin/bash


while [ ! -f "output/figure6.svg" ];
   do 
   echo "The file does not exist."
   make
   done