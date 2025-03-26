@echo off

SET BASEPATH=%~dp0

julia --project=%BASEPATH% --interactive --load=%BASEPATH%\revise.jl
