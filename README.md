# Shop: sell digital files with Stripe and no platform fees

A small, self-hosted store for selling digital downloads such as ebooks, templates, presets,
stock photos, fonts, courses and source code. Gumroad-style platforms take about 10% of each
sale. Here you pay only Stripe's processing fee, and the money goes straight to your Stripe account.

**How it works:** storefront → Stripe Checkout → order recorded (by webhook *and* the
success page, idempotently) → buyer gets a secret download link that expires and has a
download limit. The link is shown on screen and, if SMTP is set, emailed to the buyer.

## Quick start (about 10 minutes)

```bash
pip install -r requirements.txt
cp .env.example .env          # put in your Stripe test key
set -a; . ./.env; set +a
uvicorn shop.main:app --reload
```

In a second terminal, forward Stripe webhooks to your machine:

```bash
stripe listen --forward-to localhost:8000/webhook   # prints whsec_... → STRIPE_WEBHOOK_SECRET
```

Open http://localhost:8000 and buy with test card `4242 4242 4242 4242`.

## Add your products

Put files in `products/files/` and list them in `products/catalog.json`:

```json
{ "id": "lightroom-presets", "name": "50 Film Presets", "description": "…",
  "price_cents": 1900, "currency": "usd", "file": "files/presets.zip" }
```

The app checks the catalog at startup: missing files, paths outside `products/`,
and prices below Stripe's minimum all stop the app from starting.

## Go live

1. **Easiest:** on [Render](https://render.com), choose New → Blueprint and pick this repo. `render.yaml`
   sets everything up (about $7/mo for the plan with a persistent disk). Render prompts you for the Stripe keys and `BASE_URL`
   (your `https://….onrender.com` URL). Any Docker host works too; just mount `/data` so `orders.db` survives restarts.
2. Set `BASE_URL` to your public HTTPS URL and use your **live** `sk_live_…` key.
3. In Stripe Dashboard → Webhooks, add `https://your-domain/webhook` with the events
   `checkout.session.completed` and `checkout.session.async_payment_succeeded`. Put its signing
   secret in `STRIPE_WEBHOOK_SECRET`.
4. Check sales: `curl -H "Authorization: Bearer $ADMIN_TOKEN" https://your-domain/admin/orders`

Promo codes made in the Stripe Dashboard work at checkout automatically.

## The code is the easy part

What earns money is **a product people want and a way for them to find it.** Some ideas that
sell well as downloads: niche Notion/Excel templates, Lightroom presets, printable planners,
design assets, starter code for a framework you know well, and short guides on a skill you have.
Post free samples where your buyers already gather (Reddit, YouTube, X, niche forums) and link
back to the store.

## Tests

```bash
pip install -r requirements-dev.txt && pytest -q
```
