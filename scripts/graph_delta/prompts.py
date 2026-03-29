# scripts/graph_delta/prompts.py
def deltas_to_sd_prompt_remove(obj_cls: str) -> str:
    return (
        f"realistic photo of the same scene. remove the {obj_cls} in the masked region. "
        "reconstruct the background naturally (table surface / wood grain / lighting). "
        "do not change anything outside the mask."
    )


def build_add_prompt(tgt: str, category: str, extra: str = "") -> str:
    t = (tgt or "object").lower().strip()
    base = (
        "realistic photo of the same scene. top-down view. "
        f"inside the masked region, add exactly ONE {t}. "
        "match perspective, scale, lighting, and add a subtle contact shadow. "
        "keep the wooden table texture sharp. "
        "do not change anything outside the mask. no text, no logo. "
    )
    if category == "slender":
        base += (
            "the object should be a slender handheld utensil/tool with a clear handle and a functional head; "
            "do not turn it into a container or dish. "
        )
    elif category == "container":
        base += (
            "the object should be a container/vessel with an inner cavity; "
            "it should look like a real physical container sitting on the table. "
        )
    else:
        base += "the object should look like a real physical object placed on the table. "
    if extra:
        base += str(extra).strip() + " "
    return base.strip()


def build_add_negative(tgt: str, src: str = "", category: str = "generic", base_neg: str = "") -> str:
    t = (tgt or "").lower().strip()
    s = (src or "").lower().strip()
    neg = (base_neg or "")
    neg += ", clipart, cartoon, illustration, icon, sticker, 3d render"
    neg += ", frame, border, outline, halo, watermark, text, logo"
    neg += ", blurry, low detail, warped, melted, deformed, overexposed, oversmoothed, plastic look"
    neg += ", multiple, two, three, many, duplicate, duplicates, repeated, extra object"
    if s:
        neg += f", {s}"
    if category == "slender":
        neg += ", bowl, dish, plate, saucer, tray, cup, ramekin, container, vessel"
    elif category == "container":
        neg += ", fork, spoon, knife, chopsticks, blade, dagger, sword, cutlery"
    return neg
