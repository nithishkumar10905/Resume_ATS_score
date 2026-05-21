@echo off
echo Cleaning up unnecessary files...

REM Delete enhanced version files (not working due to dependencies)
del app_enhanced.py
del semantic_matcher.py
del resume_parser.py
del utils.py
del report_generator.py
del skill_ontology.yaml
del test_app.py

REM Delete extra documentation files
del README_ENHANCED.md
del API_DOCUMENTATION.md
del DEPLOYMENT.md
del MIGRATION_GUIDE.md
del PROJECT_SUMMARY.md
del PROJECT_STRUCTURE.md
del GETTING_STARTED.md
del IMPLEMENTATION_COMPLETE.md

REM Delete Docker files (not needed for local use)
del Dockerfile
del docker-compose.yml

REM Delete extra setup files
del setup.sh
del .env.example
del .gitignore

REM Delete extra requirements
del requirements_simple.txt

REM Delete app_simple.py if exists
del app_simple.py

REM Delete empty directories
rmdir /s /q cache 2>nul
rmdir /s /q reports 2>nul
rmdir /s /q logs 2>nul

echo.
echo Cleanup complete!
echo.
echo Remaining files:
echo - app.py (main application)
echo - requirements.txt (dependencies)
echo - README.md (documentation)
echo - templates/ (HTML files)
echo - static/ (CSS files)
echo - venv/ (virtual environment)
echo.
pause
