"""Disease library for the Plant Disease Recognition System.

One entry per model class (38 total). Content is general plant-health
guidance — symptoms, cultural management and prevention — written for
farmers and gardeners. No specific chemical product or dosage is
recommended here: for sprays, always ask your local agriculture
extension office for approved options in your area.
"""

DISEASE_INFO = {
    "Apple___Apple_scab": {
        "crop": "Apple",
        "label": "Apple Scab",
        "status": "diseased",
        "symptoms": "Olive-green to black velvety spots on leaves and fruit; "
                    "severe infections cause yellowing, leaf drop and cracked fruit.",
        "management": "Remove and destroy fallen leaves and infected fruit; "
                      "prune to open the canopy for airflow and faster drying.",
        "prevention": "Plant resistant varieties; avoid overhead watering; "
                      "keep the orchard floor clean in autumn.",
    },
    "Apple___Black_rot": {
        "crop": "Apple",
        "label": "Black Rot",
        "status": "diseased",
        "symptoms": "Brown leaf spots with dark borders ('frogeye' spots); "
                    "fruit rots with concentric rings; cankers on branches.",
        "management": "Remove mummified fruit and infected branches; prune "
                      "cankers back to healthy wood and destroy the prunings.",
        "prevention": "Orchard sanitation; good airflow; remove dead wood "
                      "every winter.",
    },
    "Apple___Cedar_apple_rust": {
        "crop": "Apple",
        "label": "Cedar Apple Rust",
        "status": "diseased",
        "symptoms": "Bright orange-yellow spots on leaves; in spring, jelly-like "
                    "orange galls appear on nearby juniper/cedar trees.",
        "management": "Remove galls from nearby juniper trees where practical; "
                      "pick off badly spotted leaves.",
        "prevention": "Plant resistant apple varieties; keep distance from "
                      "juniper hosts when planning a new orchard.",
    },
    "Apple___healthy": {
        "crop": "Apple",
        "label": "Healthy",
        "status": "healthy",
        "symptoms": "No disease symptoms — leaves are green, evenly coloured "
                    "and free of spots.",
        "management": "No treatment needed.",
        "prevention": "Keep monitoring weekly; maintain pruning, watering at "
                      "the base and orchard sanitation.",
    },
    "Blueberry___healthy": {
        "crop": "Blueberry",
        "label": "Healthy",
        "status": "healthy",
        "symptoms": "No disease symptoms — leaves are green and free of spots.",
        "management": "No treatment needed.",
        "prevention": "Maintain acidic, well-drained soil; mulch; monitor weekly.",
    },
    "Cherry_(including_sour)___Powdery_mildew": {
        "crop": "Cherry",
        "label": "Powdery Mildew",
        "status": "diseased",
        "symptoms": "White powdery coating on leaves and young shoots; "
                    "leaves may curl or distort.",
        "management": "Prune out affected shoots; improve airflow and sunlight "
                      "into the canopy.",
        "prevention": "Space trees well; plant resistant varieties; avoid "
                      "excess nitrogen fertiliser.",
    },
    "Cherry_(including_sour)___healthy": {
        "crop": "Cherry",
        "label": "Healthy",
        "status": "healthy",
        "symptoms": "No disease symptoms — leaves are green and free of "
                    "powdery coating or spots.",
        "management": "No treatment needed.",
        "prevention": "Prune for airflow; monitor weekly during humid weather.",
    },
    "Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot": {
        "crop": "Corn (Maize)",
        "label": "Gray Leaf Spot",
        "status": "diseased",
        "symptoms": "Rectangular gray-brown lesions between the leaf veins; "
                    "lesions can join and kill large leaf areas.",
        "management": "Remove crop residue after harvest; rotate with a "
                      "non-host crop next season.",
        "prevention": "Plant resistant hybrids; avoid dense planting; "
                      "keep the field free of maize residue.",
    },
    "Corn_(maize)___Common_rust_": {
        "crop": "Corn (Maize)",
        "label": "Common Rust",
        "status": "diseased",
        "symptoms": "Small reddish-brown pustules scattered on both leaf "
                    "surfaces; heavy infections yellow the leaves.",
        "management": "Remove heavily infected lower leaves; the disease often "
                      "slows as plants mature.",
        "prevention": "Plant resistant hybrids; sow early where the season allows.",
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "crop": "Corn (Maize)",
        "label": "Northern Leaf Blight",
        "status": "diseased",
        "symptoms": "Long, cigar-shaped gray-green lesions along the leaves; "
                    "lesions dry out and turn brown.",
        "management": "Plough in or remove infected residue; rotate crops.",
        "prevention": "Plant resistant hybrids; avoid planting maize after maize.",
    },
    "Corn_(maize)___healthy": {
        "crop": "Corn (Maize)",
        "label": "Healthy",
        "status": "healthy",
        "symptoms": "No disease symptoms — leaves are green and free of "
                    "lesions or rust pustules.",
        "management": "No treatment needed.",
        "prevention": "Rotate crops; monitor weekly, especially in humid weather.",
    },
    "Grape___Black_rot": {
        "crop": "Grape",
        "label": "Black Rot",
        "status": "diseased",
        "symptoms": "Brown circular leaf spots with tiny black dots; berries "
                    "shrink and turn into hard black mummies.",
        "management": "Remove and destroy mummified berries and infected leaves; "
                      "prune for airflow.",
        "prevention": "Clean up all fruit and leaves after harvest; train vines "
                      "for good air circulation.",
    },
    "Grape___Esca_(Black_Measles)": {
        "crop": "Grape",
        "label": "Esca (Black Measles)",
        "status": "diseased",
        "symptoms": "Tiger-stripe yellowing and browning between leaf veins; "
                    "dark spots on berries; wood decay inside the vine.",
        "management": "Prune out dead wood; protect large pruning wounds; "
                      "remove badly affected vines.",
        "prevention": "Use clean, sharp pruning tools; avoid wounding the vine; "
                      "plant certified healthy vines.",
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "crop": "Grape",
        "label": "Leaf Blight (Isariopsis Leaf Spot)",
        "status": "diseased",
        "symptoms": "Brown spots with dark borders on the leaves, sometimes "
                    "with a yellow halo.",
        "management": "Remove infected leaves; improve airflow by pruning.",
        "prevention": "Good vineyard sanitation; avoid overhead irrigation.",
    },
    "Grape___healthy": {
        "crop": "Grape",
        "label": "Healthy",
        "status": "healthy",
        "symptoms": "No disease symptoms — leaves are green and free of spots.",
        "management": "No treatment needed.",
        "prevention": "Prune for airflow; monitor weekly during wet weather.",
    },
    "Orange___Haunglongbing_(Citrus_greening)": {
        "crop": "Orange",
        "label": "Citrus Greening (Huanglongbing)",
        "status": "diseased",
        "symptoms": "Blotchy yellow mottling on leaves; small, misshapen, "
                    "bitter fruit; branch dieback. A serious, incurable disease.",
        "management": "Remove infected trees to protect healthy ones; control "
                      "the psyllid insect that spreads it.",
        "prevention": "Plant certified disease-free trees; control psyllids; "
                      "report suspected cases to your agriculture department.",
    },
    "Peach___Bacterial_spot": {
        "crop": "Peach",
        "label": "Bacterial Spot",
        "status": "diseased",
        "symptoms": "Small dark angular spots on leaves that can tear into "
                    "'shot holes'; spots and cracks on fruit.",
        "management": "Prune out infected twigs; avoid working with wet trees.",
        "prevention": "Plant resistant varieties; avoid overhead irrigation; "
                      "ask local extension about approved protective sprays.",
    },
    "Peach___healthy": {
        "crop": "Peach",
        "label": "Healthy",
        "status": "healthy",
        "symptoms": "No disease symptoms — leaves are green and free of spots.",
        "management": "No treatment needed.",
        "prevention": "Prune for airflow; monitor weekly in warm, wet weather.",
    },
    "Pepper,_bell___Bacterial_spot": {
        "crop": "Pepper (Bell)",
        "label": "Bacterial Spot",
        "status": "diseased",
        "symptoms": "Small water-soaked spots on leaves turning brown with "
                    "yellow halos; raised spots on fruit.",
        "management": "Remove infected leaves and plants; do not handle plants "
                      "when they are wet.",
        "prevention": "Use certified seed; rotate crops; water at the base "
                      "(drip irrigation); space plants for airflow.",
    },
    "Pepper,_bell___healthy": {
        "crop": "Pepper (Bell)",
        "label": "Healthy",
        "status": "healthy",
        "symptoms": "No disease symptoms — leaves are green and free of spots.",
        "management": "No treatment needed.",
        "prevention": "Rotate crops; water at the base; monitor weekly.",
    },
}

DISEASE_INFO.update({
    "Potato___Early_blight": {
        "crop": "Potato",
        "label": "Early Blight",
        "status": "diseased",
        "symptoms": "Brown spots with concentric rings ('target spots') on "
                    "older leaves; stems can show dark lesions.",
        "management": "Remove infected lower leaves; destroy badly affected "
                      "plants; rotate the plot next season.",
        "prevention": "Keep plants well fed and watered; avoid overhead "
                      "watering; plant resistant varieties.",
    },
    "Potato___Late_blight": {
        "crop": "Potato",
        "label": "Late Blight",
        "status": "diseased",
        "symptoms": "Dark water-soaked lesions on leaves with white mould on "
                    "the underside in humid weather; tubers can rot. Spreads fast.",
        "management": "Destroy infected plants at once (do not compost them); "
                      "stop overhead watering immediately.",
        "prevention": "Use certified seed tubers; space plants well; watch "
                      "closely in cool, wet weather and act at the first sign.",
    },
    "Potato___healthy": {
        "crop": "Potato",
        "label": "Healthy",
        "status": "healthy",
        "symptoms": "No disease symptoms — leaves are green and free of spots.",
        "management": "No treatment needed.",
        "prevention": "Use certified seed; rotate plots; monitor weekly.",
    },
    "Raspberry___healthy": {
        "crop": "Raspberry",
        "label": "Healthy",
        "status": "healthy",
        "symptoms": "No disease symptoms — leaves are green and free of spots.",
        "management": "No treatment needed.",
        "prevention": "Prune for airflow; mulch; monitor weekly.",
    },
    "Soybean___healthy": {
        "crop": "Soybean",
        "label": "Healthy",
        "status": "healthy",
        "symptoms": "No disease symptoms — leaves are green and free of spots.",
        "management": "No treatment needed.",
        "prevention": "Rotate crops; monitor weekly during humid weather.",
    },
    "Squash___Powdery_mildew": {
        "crop": "Squash",
        "label": "Powdery Mildew",
        "status": "diseased",
        "symptoms": "White powdery patches on leaves and stems; leaves may "
                    "yellow and dry out.",
        "management": "Remove badly affected leaves; improve airflow and sunlight.",
        "prevention": "Space plants well; water at the base; plant resistant "
                      "varieties; avoid excess nitrogen.",
    },
    "Strawberry___Leaf_scorch": {
        "crop": "Strawberry",
        "label": "Leaf Scorch",
        "status": "diseased",
        "symptoms": "Purple-brown spots on leaves; leaf edges dry and look "
                    "scorched in severe cases.",
        "management": "Remove affected leaves; keep beds clean.",
        "prevention": "Space plants for airflow; use drip irrigation and mulch; "
                      "plant certified runners.",
    },
    "Strawberry___healthy": {
        "crop": "Strawberry",
        "label": "Healthy",
        "status": "healthy",
        "symptoms": "No disease symptoms — leaves are green and free of spots.",
        "management": "No treatment needed.",
        "prevention": "Mulch; water at the base; monitor weekly.",
    },
    "Tomato___Bacterial_spot": {
        "crop": "Tomato",
        "label": "Bacterial Spot",
        "status": "diseased",
        "symptoms": "Small dark spots with yellow halos on leaves; raised "
                    "scabby spots on fruit.",
        "management": "Remove infected leaves; avoid handling wet plants.",
        "prevention": "Use certified seed; rotate crops; drip irrigation; "
                      "space plants for airflow.",
    },
    "Tomato___Early_blight": {
        "crop": "Tomato",
        "label": "Early Blight",
        "status": "diseased",
        "symptoms": "Brown spots with concentric rings on older, lower leaves; "
                    "dark lesions on stems.",
        "management": "Remove infected lower leaves; mulch under plants; "
                      "stake plants to keep leaves off the soil.",
        "prevention": "Rotate crops; water at the base; space plants well.",
    },
    "Tomato___Late_blight": {
        "crop": "Tomato",
        "label": "Late Blight",
        "status": "diseased",
        "symptoms": "Dark water-soaked lesions on leaves with white mould "
                    "underneath; rapid spread in cool, wet weather; fruit rots.",
        "management": "Remove and destroy infected plants at once; do not compost.",
        "prevention": "Use certified plants; space well; avoid overhead watering; "
                      "check daily in cool, wet spells.",
    },
    "Tomato___Leaf_Mold": {
        "crop": "Tomato",
        "label": "Leaf Mold",
        "status": "diseased",
        "symptoms": "Pale yellow spots on the upper leaf surface with olive-green "
                    "mould on the underside; common in humid greenhouses.",
        "management": "Improve ventilation and reduce humidity; remove affected "
                      "leaves.",
        "prevention": "Space plants well; ventilate greenhouses; water in the "
                      "morning so leaves dry.",
    },
    "Tomato___Septoria_leaf_spot": {
        "crop": "Tomato",
        "label": "Septoria Leaf Spot",
        "status": "diseased",
        "symptoms": "Many small circular spots with gray centres and dark "
                    "borders, mostly on lower leaves.",
        "management": "Remove infected lower leaves; mulch to stop soil splash.",
        "prevention": "Rotate crops; water at the base; stake plants.",
    },
    "Tomato___Spider_mites_Two-spotted_spider_mite": {
        "crop": "Tomato",
        "label": "Two-Spotted Spider Mite",
        "status": "diseased",
        "symptoms": "Fine webbing under leaves; tiny yellow speckles that turn "
                    "to bronzing as mites feed. A pest, not a fungus.",
        "management": "Spray leaves with water to dislodge mites; encourage "
                      "natural predators such as ladybirds.",
        "prevention": "Avoid hot, dusty conditions; check leaf undersides weekly; "
                      "keep plants well watered.",
    },
    "Tomato___Target_Spot": {
        "crop": "Tomato",
        "label": "Target Spot",
        "status": "diseased",
        "symptoms": "Brown spots with concentric rings on leaves, stems and "
                    "fruit; spots can join and defoliate the plant.",
        "management": "Remove infected leaves and fruit; improve airflow.",
        "prevention": "Rotate crops; space plants; water at the base.",
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "crop": "Tomato",
        "label": "Tomato Yellow Leaf Curl Virus",
        "status": "diseased",
        "symptoms": "Leaf edges turn yellow and curl upward; plants are stunted "
                    "with few small fruit. Spread by whiteflies.",
        "management": "Remove infected plants; control whiteflies with sticky "
                      "traps and netting.",
        "prevention": "Use insect-proof netting; plant resistant varieties; "
                      "remove weeds that host whiteflies.",
    },
    "Tomato___Tomato_mosaic_virus": {
        "crop": "Tomato",
        "label": "Tomato Mosaic Virus",
        "status": "diseased",
        "symptoms": "Mottled light and dark green mosaic pattern on leaves; "
                    "leaves may be distorted or fern-like.",
        "management": "Remove infected plants; disinfect hands and tools after "
                      "touching them.",
        "prevention": "Wash hands and tools; avoid tobacco near tomato plants; "
                      "plant resistant varieties.",
    },
    "Tomato___healthy": {
        "crop": "Tomato",
        "label": "Healthy",
        "status": "healthy",
        "symptoms": "No disease symptoms — leaves are green, evenly coloured "
                    "and free of spots.",
        "management": "No treatment needed.",
        "prevention": "Rotate crops; water at the base; monitor weekly.",
    },
})


# ---------------------------------------------------------------------------
# Seasonal guidance (general, South Asia). Qualitative only — these are
# typical patterns, not measured data, so they are shown as guidance.
# ---------------------------------------------------------------------------

SEASONAL_NOTES = {
    1: ("Cool and dry. Fungal pressure is low; watch for powdery mildew in "
        "greenhouses and rusts on cereals. Good month for pruning and sanitation."),
    2: ("Cool and dry. Early blights and leaf spots start appearing as days "
        "warm up. Keep fields clean and monitor new growth."),
    3: ("Warming up. Powdery mildew and rusts become active. Bacterial spots "
        "appear where spring rain wets the foliage."),
    4: ("Warm and dry. Spider mites thrive in hot, dusty weather — check leaf "
        "undersides. Virus-carrying whiteflies become active."),
    5: ("Hot and dry. Mites and viruses peak; fungal diseases stay low except "
        "where irrigation keeps leaves wet."),
    6: ("Pre-monsoon heat and humidity. Early blight and bacterial spots rise; "
        "keep leaves dry and improve airflow."),
    7: ("Monsoon. Highest fungal pressure of the year — late blight, leaf spots "
        "and fruit rots spread fast in warm, wet weather. Inspect daily."),
    8: ("Monsoon continues. Late blight and bacterial diseases peak. Avoid "
        "overhead watering; destroy infected plants quickly."),
    9: ("Late monsoon. Fungal pressure stays high while it is humid, then falls "
        "as weather dries. Keep up sanitation."),
    10: ("Cooling down. Leaf spots and mildews linger on late crops; good time "
        "for soil cleanup and rotation planning."),
    11: ("Cool and dry. Disease pressure falls; focus on removing crop residue "
        "and preparing beds for winter crops."),
    12: ("Cool and dry. Lowest disease pressure. Prune orchards, clean fields "
        "and plan resistant varieties for next season."),
}

SEASONAL_ACTIVE = {
    1: ["Powdery Mildew (greenhouse)", "Rusts"],
    2: ["Early Blight", "Leaf Spot"],
    3: ["Powdery Mildew", "Rusts", "Bacterial Spot"],
    4: ["Spider Mites", "Viral diseases (whitefly)"],
    5: ["Spider Mites", "Viral diseases (whitefly)"],
    6: ["Early Blight", "Bacterial Spot"],
    7: ["Late Blight", "Leaf Spot", "Fruit Rots"],
    8: ["Late Blight", "Bacterial Spot", "Leaf Spot"],
    9: ["Leaf Spot", "Late Blight"],
    10: ["Powdery Mildew", "Leaf Spot"],
    11: ["— (low pressure)"],
    12: ["— (low pressure)"],
}
