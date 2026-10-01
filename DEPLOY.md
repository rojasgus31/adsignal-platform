# Getting AdSignal live

Three services, all free tier, no card needed. Do them in this order: each
step produces a value the next one needs.

Nothing in this file is a secret. Every secret is typed straight into a
provider's dashboard and never into a file in this repository.

---

## 1. Neon — the database (about 5 minutes)

Render's own free Postgres expires after 30 days, which would take the
dashboard down a month after anyone looked at it. Neon's free tier does not
expire, so the database lives there.

1. Go to **https://neon.tech** and sign up with GitHub.
2. Create a project. Name it `adsignal`. Any region is fine; pick the one
   nearest you.
3. On the project dashboard, find **Connection string** and copy it. It looks
   like:

   ```
   postgresql://USER:PASSWORD@ep-something-123456.us-east-2.aws.neon.tech/neondb?sslmode=require
   ```

   Keep this. It is `ADSIGNAL_POSTGRES_DSN`. **This is the admin DSN** — the
   pipeline writes with it.

4. Make the read-only DSN. In the Neon SQL editor, run:

   ```sql
   CREATE ROLE adsignal_ro WITH LOGIN PASSWORD 'pick-a-long-random-password';
   ```

   Then take the connection string from step 3 and swap the user and password
   for `adsignal_ro` and the password you just chose. That is
   `ADSIGNAL_POSTGRES_READONLY_DSN`.

   This role is what `/ask` runs model-written SQL as. The nightly job grants
   it SELECT on seven tables and revokes everything on the two holding model
   artefacts and embeddings, so a generated query cannot read them even if it
   tries.

---

## 2. GitHub — fill the database, nightly and now (about 5 minutes)

The repository is already pushed. It needs three secrets so the nightly
pipeline can run.

1. Go to **https://github.com/rojasgus31/adsignal-platform/settings/secrets/actions**
2. Click **New repository secret** three times:

   | Name | Value |
   |---|---|
   | `ADSIGNAL_POSTGRES_DSN` | the admin string from step 1.3 |
   | `ADSIGNAL_POSTGRES_READONLY_DSN` | the read-only string from step 1.4 |
   | `ADSIGNAL_GEMINI_API_KEY` | your Google AI Studio key |

3. Go to the **Actions** tab, pick **Nightly pipeline**, and click
   **Run workflow**. This is what puts data in the database; until it
   finishes the dashboard has nothing to show.

   It takes 20 to 40 minutes: it generates two years of events, builds the
   mart, trains both models, scores every day and writes the embeddings. Watch
   it finish before moving on.

---

## 3. Render — the API (about 5 minutes)

1. Go to **https://render.com** and sign up with GitHub.
2. **New** → **Blueprint**. Pick `rojasgus31/adsignal-platform`. Render reads
   `render.yaml` and proposes one web service.
3. It will prompt for the four values marked `sync: false`:

   | Key | Value |
   |---|---|
   | `ADSIGNAL_POSTGRES_DSN` | admin string from step 1.3 |
   | `ADSIGNAL_POSTGRES_READONLY_DSN` | read-only string from step 1.4 |
   | `ADSIGNAL_GEMINI_API_KEY` | your Google AI Studio key |
   | `ADSIGNAL_CORS_ORIGINS` | your portfolio's origin, e.g. `https://gustavorojas.dev` |

   `ADSIGNAL_CORS_ORIGINS` must be the exact origin the dashboard is served
   from, scheme included and no trailing slash. A wildcard is rejected by the
   config on purpose. Get it wrong and every request from the dashboard is
   blocked by the browser, which looks exactly like the API being down.

4. Deploy. The first build takes about 10 minutes, mostly baking the embedding
   weights into the image.
5. Copy the service URL, e.g. `https://adsignal-api.onrender.com`. Check it:

   ```bash
   curl https://adsignal-api.onrender.com/health
   ```

   Expect `{"status":"ok"}`.

---

## 4. Vercel — point the dashboard at it (about 2 minutes)

The portfolio already deploys from `rojasgus31/portfolio`. It needs one
environment variable.

1. Vercel → your portfolio project → **Settings** → **Environment Variables**.
2. Add `PUBLIC_ADSIGNAL_API` with the Render URL from step 3.5, for
   Production. No trailing slash.
3. Redeploy.

---

## What to expect once it is live

**The first visit of the day is slow.** Render's free instance sleeps after
15 minutes idle and takes about a minute to wake. The dashboard renders that
as "waking the service up" rather than an error, which is why this trade is
acceptable here. A recruiter arriving cold will wait; that is the cost of $0.

**Ask takes a few seconds.** It calls Gemini, which usually answers in 5 to
7 seconds, and retries once if that attempt exceeds 12. Roughly one question
in six still fails; clicking again is the fix.

**The free Gemini tier has a daily quota.** If it runs out, the briefings fall
back to a local writer that composes the same paragraph deterministically, and
`/ask` refuses with a clear reason rather than breaking.

---

## Rotating the Gemini key later

Worth doing. Revoke the old key at **https://aistudio.google.com/apikey**,
create a new one, and update it in exactly two places: the GitHub Actions
secret (step 2.2) and the Render environment variable (step 3.3). No code
changes, no redeploy of the portfolio.
