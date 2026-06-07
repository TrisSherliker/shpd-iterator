# Shattered Pixel Dungeon seed iterator

An experiment using Android Developer Bridge `adb` to control a USB-connected phone. It controls a game by automated tapping, which is a useful low-risk playground. Tap, wait, tap, wait...

This script depends on [Shattered Pixel Dungeon](https://github.com/00-Evan/shattered-pixel-dungeon) installed on Android and [Elektrochecker's seed finder](https://github.com/Elektrochecker/shpd-seed-finder) on a local machine.  

Highly unportable: the python script has hardcoded locations for OS calls and tap coordinates, specific to the PC and phone being used. 