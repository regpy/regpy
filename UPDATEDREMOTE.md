# WARNING the project moved

In the process to optimize and improve the structure of the repository we decided that, for the internal structure on GitLab, we created a new Group RegPy which now holds the repository of the same name. Thus, the repository path has changed from: `cruegge/itreg` to `regpy/regpy`. This Change requires you to update the remote-url in your local repositories. Note that if you do not change the url you push to the wrong repository!

To simplify the migration, I have prepared small update scripts that automatically adjust your Git remote configuration:

- **For Linux/macOS (Git Bash):** run the `update-remote.sh` script in the root of your local repository:

    ```./update-remote.sh``` 

    In case the script is not executable please first run 

    ```chmod +x update-remote.sh```

- **For Windows (CMD):** run the `update-remote.cmd` script in the root of your local repository:

    ```update-remote.cmd```

These scripts will detect whether your remote is configured via HTTPS or SSH and update it accordingly.

- **Manual Update (alternative)** - Use only if the script is not working. If you prefer not to use the scripts, you can update the remote manually:
  - For SSH remotes:

    ```git remote set-url origin git@gitlab.com:regpy/regpy.git```

  - For HTTPS remotes:

    ```git remote set-url origin https://gitlab.com/regpy/regpy.git```

Please ensure you run the script in every local clone of the repository you are working with.