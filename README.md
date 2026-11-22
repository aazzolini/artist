# Shop: sell 3D-printed figures of your characters (and digital downloads)

A small, self-hosted store. You list your characters, buyers pay with Stripe, and each
order goes to a print-on-demand service that prints the figure and ships it to the buyer.
You never hold inventory. Payments go straight to your Stripe account; there's no marketplace cut.

**How an order works**

1. The buyer picks a figure and pays on Stripe Checkout. Stripe collects a US shipping address and charges a flat shipping fee.
2. The order is recorded exactly once, by Stripe's webhook and the success page.
3. It's handed to the **fulfiller**. Today that means you get an email with the address and the
   STL model attached, which you place with your print service. The buyer gets a confirmation and
   an order-tracking page.
4. When it ships, you mark it shipped with a tracking number and the buyer is emailed.

Digital downloads are supported too: set `"kind": "digital"` and buyers get an expiring download link.

## From your art to a product

Each figure needs two files:

- **An image** for the store (`products/images/`): the character art you already have.
- **A 3D model** (`products/files/*.stl`): your 2D art turned into a printable model. AI
  image-to-3D tools (Meshy, Tripo and similar) do this from a single image. Check that the model
  is watertight and has a flat base, or ask your print service to check it.

Then add it to `products/catalog.json`:

```json
{ "id": "neon-fox", "kind": "physical", "name": "Neon Fox Figure",
  "description": "10 cm figure of our Neon Fox character, printed to order.",
  "price_cents": 3900, "currency": "usd",
  "file": "files/neon-fox.stl", "image": "neon-fox.png" }
```

The app checks the catalog at startup. Missing files, paths outside `products/` and prices below
Stripe's minimum all stop the app from starting. Images are public; model files never are.

**Rights:** sell only characters you own. Don't sell figures of real people, such as public figures
from news art, or of other companies' characters. Check your AI image tool's terms for commercial use.

## Run it locally

```bash
pip install -r requirements.txt
cp .env.example .env          # put in your Stripe test key
set -a; . ./.env; set +a
uvicorn shop.main:app --reload
stripe listen --forward-to localhost:8000/webhook   # second terminal; prints STRIPE_WEBHOOK_SECRET
```

Open http://localhost:8000 and buy with test card `4242 4242 4242 4242`.

## Go live

1. **Stripe:** create an account (free), finish identity and bank verification, and copy the secret key.
2. **Render:** choose New → Blueprint and pick this repo. `render.yaml` sets everything up (about $7/mo
   for the plan with a persistent disk). Fill in the prompted values; `BASE_URL` is your
   `https://….onrender.com` URL. Any Docker host also works if you mount `/data`.
3. **Stripe webhook:** add `https://your-domain/webhook` for `checkout.session.completed` and
   `checkout.session.async_payment_succeeded`, then put its signing secret in `STRIPE_WEBHOOK_SECRET`.
4. **Email** (needed for order emails): any SMTP provider, for example your email host or Resend/Postmark/SES SMTP.
   Set `SMTP_*` and `OWNER_EMAIL`.
5. **Print service:** open an account with a print-on-demand 3D printing service that ships to
   customers (for example Slant 3D). Price figures above print cost + shipping + Stripe's fee.

## Running the shop

```bash
# Orders still to place with the printer
curl -H "Authorization: Bearer $ADMIN_TOKEN" "https://your-domain/admin/orders?status=awaiting_fulfillment"
# All orders + revenue
curl -H "Authorization: Bearer $ADMIN_TOKEN" https://your-domain/admin/orders
# Mark shipped (emails the buyer)
curl -X POST -H "Authorization: Bearer $ADMIN_TOKEN" -H "Content-Type: application/json" \
     -d '{"tracking":"1Z..."}' https://your-domain/admin/orders/<session_id>/ship
```

Order statuses: `awaiting_fulfillment` (not handed off yet, e.g. email failed) → `submitted` → `shipped`.

**Automating the printer:** fulfillment lives in `shop/fulfillment.py`. To send orders straight to a
print service's API instead of emailing you, add a class with a `submit(order, product)` method
that calls that API and returns its order ID, and select it in `make_fulfiller`.

## Tests

```bash
pip install -r requirements-dev.txt && pytest -q
```

## Relief artwork: printing News in Letters pieces in color and depth

`relief/` turns a News in Letters drawing script into a full-color, high-relief plaque.
The picture is made of raised letters, and each object in the scene sits at its own depth.

```bash
pip install -r requirements-dev.txt
python -m relief.build 2026-11-16-15 --width-mm 200 --cols 133 --out out/
# -> out/2026-11-16-15-133c/: .3mf and OBJ zip (colored, watertight), previews, letters.txt
```

- `relief/studio.py` and `relief/pieces/p_<id>.py` are the site's own drawing scripts. They're re-run at
  print resolution, and every layer they paint is recorded.
- `relief/pieces/s_<id>.py` sets each layer's depth (mm) and color. Adding a piece takes a drawing script
  plus this short spec.
- **Letter size:** letters about 1.5 mm wide (133 across a 20 cm plaque) are readable from 30 cm and
  have 0.4 mm strokes. Bigger plaques should get more letters, not bigger ones: for example, 200 across 30 cm.
- Output: a 3 mm base plus up to about 9.5 mm of relief. The 1.5M-triangle mesh keeps the letters to within
  0.02 mm (95th percentile). Upload the `.3mf`, or the OBJ zip if a service prefers it, as a
  **full-color PolyJet** print (Craftcloud, Xometry, Stratasys Direct, Unionfab).

Sample previews: `relief/samples/the-catch-200mm/`.
