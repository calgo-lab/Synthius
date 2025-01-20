1. **Update Version in `pyproject.toml`**:
   - Open the `pyproject.toml` file.
   - Locate the `[tool.poetry]` section.
   - Update the `version` field with the new version (e.g., `version = "0.1.2"`).


2. **Update Tag in GitHub Actions Workflow**:
   - Update the tag trigger in the `publish.yml` file.
   - Change the line under `on: push: tags:` to match the new version (e.g., `- 'v0.1.2'`).


3. **Commit the Changes**:
   - Commit changes to both `pyproject.toml` and `publish.yml` to ensure they are ready for the new release.

4. **Create a New Git Tag**:
   - Tag commit with the new version (e.g., `v0.1.2`).

   ```bash
   git tag v0.1.2
   ```

5. **Push the Tag to GitHub**:
   - Push changes and the new tag to GitHub to trigger the release workflow.

   ```bash
   git push origin main
   git push origin v0.1.2
   ```

## Remove Tags
```bash
    git tag -d v0.1.1   
    git push origin :refs/tags/v0.1.1
```