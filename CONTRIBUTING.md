<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Contributing to Duranta

We want to make contributing to this project as easy and transparent as possible.

1. Create an account on [GitHub](https://github.com/). Only contributions
   against [`duranta-project/openairinterface5g/`](https://github.com/duranta-project/openairinterface5g/)
   are accepted.
2. Fork the repository, and open pull requests for your contributions from your
   fork.
3. The contributing policies are described in the [corresponding documentation
   page](doc/code-style-contrib.md).
4. [Sign the CLA](https://github.com/duranta-project/governance/blob/main/docs/easy_cla_process.md)
   either before doing making your first pull request or after submitting the
   pull request.
5. Mandatory signing of all the commits using the email address used for CLA.

## Commit Guidelines

Every pull request must pass two CI checks before it can be merged:

1. **[Developer Certificate of Origin (DCO)](https://en.wikipedia.org/wiki/Developer_Certificate_of_Origin)**:
   Each commit must include a `Signed-off-by:` trailer in the commit message.
   Use `git commit -s` (or `--signoff`).

2. **[Verified commits](https://docs.github.com/en/authentication/managing-commit-signature-verification/about-commit-signature-verification)**:
   Each commit must be cryptographically signed using SSH or GPG keys to confirm
   its origin.

### Signing Commits

GitHub supports commit signing using either SSH keys or GPG keys.
For more information, see the
[GitHub documentation](https://docs.github.com/en/authentication/managing-commit-signature-verification/signing-commits).

Before configuring commit signing:

- Generate an SSH key pair or GPG key pair.
- Add your public key to your GitHub account.
- Verify your GitHub email address (required for “Verified” commits to work).
- If using SSH signing, ensure the key is registered in GitHub for:
    - Authentication (SSH and GPG keys)
    - Signing commits (Signing Keys)

> **NOTE:**
> Adding an SSH key for repo access does not automatically enable commit signing.
> The key must also be added under GitHub's Signing Keys settings.


To ensure commits show as Verified on GitHub:

- Your `git config user.email` must match a GitHub email
- That email must be verified in your GitHub account

For more information, see the
[GitHub Docs](https://docs.github.com/en/account-and-profile/how-tos/email-preferences/verifying-your-email-address)

Configure your repository's `.git/config`:

```ini
# Edit the git configuration

[user]
    name = YOUR NAME
    email = YOUR VERIFIED EMAIL ADDRESS

    # REQUIRED for commit signing
    # Use ONE signing method (SSH or GPG)

    signingkey = YOUR_SIGNING_KEY

    # Examples:
    # SSH signing:
    # signingkey = ~/.ssh/id_ed25519.pub

    # GPG signing:
    # signingkey = YOUR_GPG_KEY_ID

[gpg]
    # REQUIRED: defines signing method (SSH or GPG)

    format = YOUR_SIGNING_FORMAT

    # Examples:
    # SSH signing:
    # format = ssh

    # GPG signing:
    # format = openpgp

[commit]
    gpgsign = true
```

> The private key is used automatically by SSH/Git when signing commits (SSH only).

#### Verifying Signed Commits

You can verify that commits are properly signed locally using:

```bash
git log --show-signature
```

GitHub should also display a Verified badge next to signed commits once the
signing key has been correctly configured in your account.

##### SSH Signature Verification (`allowed_signers`)

For SSH commit signing, local Git verification may require an `allowed_signers`
file. This is only used for local verification in Git and is not required
by GitHub.

If you see errors such as:

```text
No principal matched
Can't check signature
error: gpg.ssh.allowedSignersFile needs to be configured
```
you may need to configure it.

Create the file and add your signing identity:

```bash
mkdir -p ~/.config/git
touch ~/.config/git/allowed_signers
echo "user@example.com ssh-ed25519 AAAACexamplekeystringhere" > ~/.config/git/allowed_signers
```

Enable it in local repository Git config:

```bash
git config gpg.ssh.allowedSignersFile ~/.config/git/allowed_signers
```

> **NOTE:**
> This is only for local Git signature verification and does not affect GitHub,
> or remote repository behavior.

> **NOTE:** If your commits are not signed, the CI framework will not accept the PR.
For more information regarding contribution guidelines
please check [this document](doc/code-style-contrib.md)

## License

By contributing to Duranta, you agree that your contributions will be
licensed under

1. [CSSL v1.0 license](LICENSES/preferred/CSSL-v1.0.txt): for RAN and UE
   related source code and test scripts
2. [CC-BY-4.0](LICENSES/preferred/CC-BY-4.0.txt): All the documentation
3. [MIT](LICENSES/preferred/MIT.txt): Orchestration (helm-charts, docker
   compose)

Certain files are using different licenses; you can read about them in
[NOTICE](NOTICE).
