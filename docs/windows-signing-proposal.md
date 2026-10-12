# Windows release signing proposal

Evaluated on 2026-10-12. This is a proposal; no account, certificate or purchase was created.

## Eligibility and cost

Prefer Microsoft Artifact Signing Basic with a Public Trust certificate profile if the
publisher is eligible. Microsoft's current [prerequisites](https://learn.microsoft.com/en-us/azure/artifact-signing/quickstart)
accept EU organizations, among other regions, but individuals must be located in the
US or Canada. Europe/Berlin timezone does not establish legal domicile or organization status.
A European individual without a qualifying organization needs another publicly trusted
code-signing provider; do not provision an ineligible Azure account.

The [SmartScreen developer guidance](https://learn.microsoft.com/en-us/windows/apps/package-and-deploy/smartscreen-reputation)
quotes a starting price of USD 9.99/month. The [pricing page](https://azure.microsoft.com/en-us/pricing/details/artifact-signing/)
lists Basic's 5,000 signatures/month, but its fetched monetary fields were unavailable.
Verify the live regional/currency quote before purchase. Two executables per release are
well within Basic's published quota. A [paid Azure subscription is required](https://learn.microsoft.com/en-us/azure/artifact-signing/faq);
free/trial/sponsored subscriptions are unsupported. Account creation starts billing,
and the monthly fee is not prorated.

## Concrete CI integration

1. Validate the publisher's legal identity and eligible country/entity type. Create a
   paid Azure account and Public Trust profile only after procurement approval.
2. Configure a GitHub protected `release-signing` environment. Federate Azure login
   through GitHub OIDC restricted to this repository and environment; assign the
   Artifact Signing Certificate Profile Signer role only on the chosen profile.
3. Enable `id-token: write` only in the signing job. Use Azure login with client, tenant
   and subscription IDs; keep no private key or client secret in the repository.
4. Use the Microsoft [Artifact Signing GitHub action](https://learn.microsoft.com/en-us/azure/artifact-signing/how-to-signing-integrations)
   pinned to a reviewed commit SHA. Configure the endpoint, account and Public Trust
   profile. Sign only `whisper-key.exe` and `whisper-key-console.exe`, with SHA256
   file/timestamp digests and RFC3161 timestamping. Exclude diagnostic executables.
5. After signing, run `signtool verify /pa /all /v` on both files, require a valid
   `Get-AuthenticodeSignature` and a timestamp certificate, and confirm the expected
   legal publisher. A signing/verification failure must fail publication.
6. Compute `SHA256SUMS.txt` after signing and preserve the signed bytes unchanged for
   scanning, workflow artifacts and GitHub releases. Include profile identity and
   verification results in build provenance, but no credentials or signed temporary URLs.

Do not enable a partially configured signing job: retain the existing unsigned pipeline
until identity, billing and OIDC/profile access are ready. Published old unsigned binaries
remain distinct samples and may need their own vendor submissions.

## Expected result

Consistent signing establishes publisher identity and can support reputation across
releases. It does not guarantee no Defender detections or immediate SmartScreen trust.
[EV no longer bypasses SmartScreen](https://learn.microsoft.com/en-us/windows/apps/package-and-deploy/smartscreen-reputation).
The application wheel checksum is independent of Authenticode; the updater still obtains
both the wheel and checksum manifest from GitHub. Signing the launchers does not sign
or authenticate the separately downloaded Python wheels.
