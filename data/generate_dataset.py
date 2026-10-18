"""Generate the sample customer-support dataset used to train the classifier.

The dataset is synthetic: it is assembled from hand-written issue sentences,
optional greetings/closings, and realistic details (order numbers, plan names,
error codes). Running this script is deterministic (fixed random seed), so the
CSV can always be regenerated exactly.

Usage:
    python data/generate_dataset.py
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

SEED = 42
SAMPLES_PER_CLASS = 150
OUTPUT_PATH = Path(__file__).with_name("support_tickets.csv")

# Hand-written issue sentences per category. Each one describes a distinct
# kind of problem a support team would see.
ISSUES: dict[str, list[str]] = {
    "billing": [
        "I was charged twice for my subscription this month",
        "my invoice shows the wrong amount",
        "I want a refund for the last payment",
        "why was my card billed after I cancelled",
        "the price on my bill is higher than the plan I signed up for",
        "I need a copy of my invoice for my accounts team",
        "there is an unexpected fee on my statement",
        "my promo code was not applied to my order total",
        "please update the billing address on my invoice",
        "I was overcharged for the extra seat we added",
        "the VAT on my receipt looks incorrect",
        "I can't find the charge for last month on my statement",
        "my payment failed but I still see a pending charge",
        "can I switch from monthly billing to annual billing",
        "I was charged for a plan I never purchased",
        "the tax amount on my order is wrong",
        "I'd like to dispute a transaction on my bank statement",
        "my discount disappeared from the renewal invoice",
        "the subscription renewed without notifying me and I was charged",
        "I need a receipt showing the amount paid for expense claims",
        "the invoice total doesn't match what I agreed to",
        "please credit my account for the days I didn't use",
        "why did my price go up at renewal",
        "I was billed in the wrong currency",
        "the late payment fee was added even though I paid on time",
    ],
    "technical_issue": [
        "the app crashes every time I open the dashboard",
        "the page keeps showing an error 500",
        "uploads fail halfway through with a timeout",
        "the export button does nothing when I click it",
        "the mobile app freezes on the home screen",
        "reports are loading forever and never finish",
        "I'm getting a blank screen after the latest update",
        "the sync between my phone and the desktop app is broken",
        "notifications stopped working after the last release",
        "the search function returns no results even for exact matches",
        "the video player won't play any recordings",
        "the API is returning a 503 error intermittently",
        "the charts on my dashboard show incorrect data",
        "file attachments are corrupted when I download them",
        "the website is extremely slow since yesterday",
        "I see a database connection error when saving my changes",
        "the integration with our calendar stopped syncing events",
        "the app logs me out every few minutes during use",
        "the checkout page throws a JavaScript error and won't load",
        "our scheduled jobs are failing with a memory error",
        "the printer option is missing from the print menu",
        "the browser extension stops responding after a few clicks",
        "the software installer fails with error code 0x80070005",
        "the layout is broken on Safari and buttons overlap",
        "data imported from CSV is showing up with missing rows",
    ],
    "account_access": [
        "I can't log in to my account",
        "I forgot my password and the reset email never arrived",
        "my account is locked after too many failed attempts",
        "two-factor codes are not being accepted on my phone",
        "I need to change the email address linked to my account",
        "how do I add a new user to our team account",
        "I was removed from my company workspace by mistake",
        "my login works on the web but not in the mobile app",
        "I'm not receiving the verification code by SMS",
        "I can't access the admin panel anymore",
        "please reactivate my account that was suspended",
        "single sign-on keeps redirecting me back to the login page",
        "I need to transfer ownership of the account to a colleague",
        "my username was changed and now I can't sign in",
        "how do I remove access for a former employee",
        "I lost my authenticator app and need to reset two-factor",
        "the invite link to join our organisation has expired",
        "my account says my permissions are read-only and I need edit access",
        "I can't verify my identity to recover my account",
        "I keep getting 'invalid credentials' even though the password is correct",
        "how can I delete my account and all associated data",
        "I can't sign in from my new laptop",
        "my session expires immediately after I log in",
        "I need to unlock access to the reporting module for my role",
        "the password policy rejects every new password I try",
    ],
    "general_enquiry": [
        "do you offer a discount for non-profit organisations",
        "what are your opening hours over the holidays",
        "can you tell me whether the product works with Linux",
        "is there a free trial available before I buy",
        "where can I find the user guide for the new features",
        "do you have an office in Europe",
        "I'd like to know if you are hiring for the support team",
        "what is the difference between the standard and pro plans",
        "can I get a demo of the product for my team next week",
        "does your service comply with GDPR",
        "are there any upcoming webinars I can attend",
        "how long does onboarding usually take",
        "do you support integrations with Slack and Teams",
        "I'd like to leave some general feedback about the product",
        "which languages does the interface support",
        "is there a partner programme for resellers",
        "could you send me the product brochure",
        "what is your policy on data retention",
        "can I speak to someone about enterprise pricing options",
        "do you publish a public roadmap for upcoming features",
        "is there a mobile app available for tablets",
        "where can I read your terms of service",
        "how do I subscribe to your newsletter",
        "is it possible to schedule a call with the sales team",
        "what kind of customer support do you offer on weekends",
    ],
}

OPENERS = [
    "",
    "Hi, ",
    "Hello there, ",
    "Hey team, ",
    "Good morning, ",
    "Hi support, ",
    "Sorry to bother you but ",
]

CLOSERS = [
    "",
    ". Thanks in advance.",
    ". Please help when you can.",
    ". This is urgent for us.",
    ". Let me know what to do next.",
    ". Thank you!",
    ". Any update would be appreciated.",
]

DETAILS = [
    "",
    " (order #{order})",
    " on my account {account}",
    " for plan {plan}",
    " since {day}",
    " with error code E{code}",
]

PLANS = ["Starter", "Team", "Business", "Enterprise", "Pro"]
DAYS = ["Monday", "Tuesday", "last week", "this morning", "yesterday", "two days ago"]


def _capitalise(text: str) -> str:
    return text[:1].upper() + text[1:]


def _fill_details(detail: str, rng: random.Random) -> str:
    return detail.format(
        order=rng.randint(10000, 99999),
        account=f"ACME-{rng.randint(100, 999)}",
        plan=rng.choice(PLANS),
        day=rng.choice(DAYS),
        code=rng.randint(100, 999),
    )


def build_rows(samples_per_class: int = SAMPLES_PER_CLASS, seed: int = SEED) -> list[dict]:
    """Return a list of {"text": ..., "label": ...} rows, shuffled deterministically."""
    rng = random.Random(seed)
    rows: list[dict] = []
    seen: set[str] = set()

    for label, issues in ISSUES.items():
        count = 0
        attempts = 0
        # Each issue sentence is used repeatedly with different wrappers so the
        # class gets enough rows; duplicates are skipped.
        while count < samples_per_class and attempts < samples_per_class * 50:
            attempts += 1
            issue = rng.choice(issues)
            opener = rng.choice(OPENERS)
            closer = rng.choice(CLOSERS)
            detail = _fill_details(rng.choice(DETAILS), rng)
            body = issue + detail
            # Lower-case the first letter when a greeting precedes the sentence,
            # but keep the pronoun "I" capitalised.
            if opener and not (body.startswith("I ") or body.startswith("I'")):
                body = body[:1].lower() + body[1:]
            text = _capitalise(opener + body) + closer
            if not closer:
                text += "."
            if text in seen:
                continue
            seen.add(text)
            rows.append({"text": text, "label": label})
            count += 1

    rng.shuffle(rows)
    return rows


def write_csv(rows: list[dict], path: Path = OUTPUT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    data = build_rows()
    write_csv(data)
    counts: dict[str, int] = {}
    for row in data:
        counts[row["label"]] = counts.get(row["label"], 0) + 1
    print(f"Wrote {len(data)} rows to {OUTPUT_PATH}")
    for label, count in counts.items():
        print(f"  {label}: {count}")
