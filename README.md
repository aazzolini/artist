## Set up (no config files)

1. **Put it online:** on [render.com](https://render.com), choose New → Blueprint and pick this repo (about $7/month).
2. **Open `https://<your-store>.onrender.com/admin`** and create your admin password. Do this right away:
   the first visitor to that page sets it.
3. **Connect payments** on the same page: paste your Stripe secret key from
   [dashboard.stripe.com/apikeys](https://dashboard.stripe.com/apikeys). Stripe holds your bank details and pays
   you out. The store checks the key and registers its order notifications with Stripe by itself.
4. **Price a piece and click Go live.** Paid orders show up under "Orders to print", with the address
   and a print-file download.

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
python -m relief.build 2026-11-16-15 --width-mm 200 --cols 133 --frame-mm 15 --out out/
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

- **Frame:** `--frame-mm 15` (the default) adds a sculpted molding around the picture, made of the same
  letters. It has a lip, a rounded crest 11 mm above the base, a cove and a bead, with mitered corners
  and gilt-to-bronze color. The letters are dense where the molding faces the light. Use `--frame-mm 0`
  for none. The frame nearly doubles the print volume (367 vs 196 cm³ for The Catch), which matters
  because full-color prints are usually priced by volume.

- **Ready to hang** (framed pieces): the back is hollowed like a real frame. It keeps a 2.5 mm shell
  behind every front surface and a solid 4 mm rim where the piece touches the wall, which cuts The Catch
  from 367 to 268 cm³. Two keyhole slots are cut into the back of the frame's side bands, 45 mm below
  the top and 217 mm apart. Each has a 9 mm entry, a 4.6 mm shank slot through a 2.4 mm lip, and a 4 mm
  deep channel behind it for the screw head. That fits #6–#8 screws with heads up to about 8.5 mm.
  Unframed pieces are too thin for keyholes, so they get none. Wall-facing surfaces print in plain
  bronze. The keyholes are cut with manifold3d booleans, and the front and back surfaces are
  simplified separately and stitched together, so the solid is always watertight.
  `info.json` records the keyhole positions, screw spacing, volume and how the mesh was simplified.

Sample previews: `relief/samples/the-catch-framed/`.

**3D web preview:** each build also writes the web model. `<name>.model.json` holds the glTF and
base64 geometry, and `<name>-0..2.png` are the color, letter-detail and gloss maps. The shape is about
200k triangles, and the raised letters come from a full-resolution normal map. `relief/viewer/index.html`
is a three.js viewer for it: orbit, zoom into the letters, move or sweep the light, and switch between
matte, satin and gloss finishes. Put the web model files next to it, named `the-catch-framed.*`, or change the
filename in the page. The page rebuilds the model in memory, so it never fetches `data:` or `blob:` URLs.
That matters because strict hosts, such as claude.ai artifacts, block those fetches. (`<name>.glb` is
also written for desktop 3D viewers.)

## Automated pipeline: from a drawing script to a product page

```
relief/pieces/p_<id>.py  (a News in Letters drawing script; optional r_<id>.json with title/headline)
   │  git push to main
   ▼
GitHub Action "Relief pieces" (.github/workflows/relief-pieces.yml), for each new or changed piece:
   1. spec      Claude reads the script plus a measured table of its layers and returns depth, shape and
                color for each layer as schema-checked JSON (relief/spec.py). No generated code runs.
                Without ANTHROPIC_API_KEY, or if Claude declines, a heuristic spec is used. A
                hand-written s_<id>.py always wins. Generated specs are committed back for review.
   2. build     framed, ready-to-hang relief: print.3mf, web 3D model, previews
   3. QA        watertight, consistent winding, every edge joining exactly two faces, size limits,
                letter size, mesh accuracy on the letters (p95 ≤ 0.1 mm), two level keyholes.
                Any failure stops the piece.
   4. price     from measured volume and pricing.json. No rates set means no price, so it's a draft.
   5. publish   bundle.zip → POST {STORE_URL}/admin/pieces (ADMIN_TOKEN)
   ▼
Store: /p/<id> product page with the 3D viewer. Drafts are hidden from the storefront and can't be
bought; the publish response includes a private preview link.
   ▼
Order paid → print-order email to OWNER_EMAIL: shipping address, print-service instructions, and a
signed 14-day link to the print file (never public). Place it with your print service, then
POST /admin/orders/<id>/ship with the tracking number, and the buyer is emailed.
```

Run it locally (same steps):

```bash
pip install -r requirements-dev.txt anthropic
ADMIN_TOKEN=... python -m relief.pipeline run 2026-11-16-15 --store https://your-store.example
python -m relief.pipeline run <id> --no-claude            # heuristic spec
python -m relief.pipeline spec <id>                       # (re)generate just the spec
python -m relief.pipeline changed --since HEAD~3          # which pieces changed
```

Owner controls on the store:

```bash
# set a price and take a piece live (or {"draft": true} to hide it)
curl -X POST -H "Authorization: Bearer $ADMIN_TOKEN" -H "Content-Type: application/json" \
     -d '{"price_cents": 18900, "draft": false}' https://your-store.example/admin/products/<id>
```

Once you have real print quotes, fill in `print_cost_per_cm3` and `print_cost_base` in `pricing.json`.
New pieces are then priced automatically. A republished piece with no price keeps the price you set.

**What isn't automated yet:** placing the order with the print service. I couldn't verify a
full-color print API that ships to customers, so `shop/fulfillment.py` emails you instead. Once you
have API access and docs from a service, a class with a `submit(order, product, print_url)` method
that calls their API slots in through `make_fulfiller`.

Repository secrets for the Action: `STORE_URL`, `ADMIN_TOKEN`, and optionally `ANTHROPIC_API_KEY`.
On the store, set `OWNER_EMAIL` and `SMTP_*` so print orders reach you.
