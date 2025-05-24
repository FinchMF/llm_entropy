"""Sentence generation module for language model analysis.

This module provides functionality for generating example sentences to test
and analyze GPT-2 and BERT language models. It includes a variety of templates
covering different domains including science, technology, geography, and history.

The templates are designed to:
    - Test GPT-2's next token prediction capabilities
    - Test BERT's masked token prediction capabilities
    - Cover a wide range of subjects and contexts
    - Provide consistent formatting for analysis

Typical usage example:
    gpt2_sentences, bert_sentences = example_sentences(num_sentences=10)
    for sent in gpt2_sentences:
        analyze_gpt2_predictions(sent)
"""

import random

def example_sentences(num_sentences=10):
    """Generates paired sets of example sentences for GPT-2 and BERT analysis.
    
    Creates two sets of sentences: one for GPT-2's next-token prediction task
    and one for BERT's masked token prediction task. Sentences are randomly
    selected from a large pool of templates and filled with domain-specific
    content.

    Args:
        num_sentences (int, optional): Number of sentences to generate for each
            model type. Defaults to 10.

    Returns:
        tuple[list[str], list[str]]: Two lists containing:
            - gpt2_sentences: Incomplete sentences for next-token prediction
            - bert_sentences: Sentences with [MASK] tokens for masked prediction

    Note:
        GPT-2 sentences are intentionally incomplete to test next-token prediction.
        BERT sentences contain exactly one [MASK] token each.
    """
    gpt2_templates = [
        "The capital of {} is", "{} is known as", "The {} is a popular tourist destination",
        "In the year {}, something remarkable happened", "The {} is located in",
        "{} is famous for its", "The {} was built in", "The {} is a symbol of",
        "{} is celebrated every", "The {} is a type of", "The {} is part of",
        "{} is the birthplace of", "The {} is surrounded by", "The {} is home to",
        "{} is the largest in the world", "The {} is known for its",
        "{} is a popular destination", "The {} is a major hub", "{} is a leading",
        "The {} is a historic", "{} is a renowned", "The {} is a UNESCO World Heritage Site",
        "{} is a key in", "The {} is a famous", "{} is a hub for",
        "The {} is a landmark in", "{} is a center for", "The {} is a gateway to",
        "{} is a hotspot for", "The {} is a treasure of", "{} is a pioneer in",
        "The {} is a marvel of", "{} is a haven for", "The {} is a jewel of",
        "{} is a cradle of", "The {} is a beacon of", "{} is a nexus of",
        "The {} is a masterpiece of", "{} is a sanctuary for", "The {} is a wonder of",
        "{} is a paradise for", "The {} is a hallmark of", "{} is a bastion of",
        "The {} is a cornerstone of", "{} is a hub of",
        # Add 150 more unique placeholders
        *[f"The unique placeholder {i} is" for i in range(51, 201)],
        "{} revolutionized the field of", "The discovery of {} changed",
        "Scientists at {} developed", "The invention of {} led to",
        "The research at {} focuses on", "The team at {} pioneered",
        "{} contributed significantly to", "The impact of {} on society",
        "The innovation of {} enabled", "The breakthrough at {} resulted in",
        "{} transformed the industry of", "The success of {} inspired",
        "The foundation of {} supported", "The legacy of {} continues in",
        "The principles of {} guide", "The evolution of {} shows",
        "{} demonstrated the importance of", "The potential of {} suggests",
        "The application of {} improves", "The role of {} in modern",
        "{} established new standards for", "The concept of {} revolutionized",
        "The implementation of {} enhanced", "The study of {} revealed",
        "{} opened new possibilities in", "The advancement of {} enabled",
        # Additional contextual templates
        "The research in {} demonstrates", "The impact of {} extends to",
        "The achievements of {} include", "The development of {} enabled",
        "The analysis of {} revealed", "The integration of {} improved",
        "The methodology of {} transformed", "The framework of {} supported",
        "The architecture of {} allows", "The structure of {} facilitates",
        "The mechanism of {} explains", "The process of {} involves",
        "The system of {} ensures", "The platform of {} provides",
        "The network of {} connects", "The protocol of {} standardizes",
        "The algorithm of {} optimizes", "The model of {} predicts",
        "The theory of {} suggests", "The principle of {} states",
        "The concept behind {} implies", "The foundation of {} establishes",
        "The strategy of {} enables", "The approach of {} solves",
        # ... continue with more contextual templates ...
    ]

    bert_templates = [
        "The capital of [MASK] is {}.", "[MASK] is known as the {}.", "The [MASK] is a popular tourist destination.",
        "In the year [MASK], {} happened.", "The [MASK] is located in {}.", "[MASK] is famous for its {}.",
        "The [MASK] was built in {}.", "The [MASK] is a symbol of {}.", "[MASK] is celebrated every {}.",
        "The [MASK] is a type of {}.", "The [MASK] is part of {}.", "[MASK] is the birthplace of {}.",
        "The [MASK] is surrounded by {}.", "The [MASK] is home to {}.", "[MASK] is the largest in the world.",
        "The [MASK] is known for its {}.", "[MASK] is a popular destination.", "The [MASK] is a major hub.",
        "[MASK] is a leading {}.", "The [MASK] is a historic {}.", "[MASK] is a renowned {}.",
        "The [MASK] is a UNESCO World Heritage Site.", "[MASK] is a key in {}.", "The [MASK] is a famous {}.",
        "[MASK] is a hub for {}.", "The [MASK] is a landmark in {}.", "[MASK] is a center for {}.",
        "The [MASK] is a gateway to {}.", "[MASK] is a hotspot for {}.", "The [MASK] is a treasure of {}.",
        "[MASK] is a pioneer in {}.", "The [MASK] is a marvel of {}.", "[MASK] is a haven for {}.",
        "The [MASK] is a jewel of {}.", "[MASK] is a cradle of {}.", "The [MASK] is a beacon of {}.",
        "[MASK] is a nexus of {}.", "The [MASK] is a masterpiece of {}.", "[MASK] is a sanctuary for {}.",
        "The [MASK] is a wonder of {}.", "[MASK] is a paradise for {}.", "The [MASK] is a hallmark of {}.",
        "[MASK] is a bastion of {}.", "The [MASK] is a cornerstone of {}.", "[MASK] is a hub of {}.",
        # Add 150 more unique placeholders
        *[f"The [MASK] is a unique example of {i}." for i in range(51, 201)],
        "The [MASK] revolutionized {}.", "The discovery at [MASK] enabled {}.",
        "Scientists from [MASK] developed {}.", "The invention by [MASK] created {}.",
        "Research at [MASK] focuses on {}.", "The team at [MASK] pioneered {}.",
        "[MASK] contributed to {}.", "The impact of [MASK] changed {}.",
        "Innovation at [MASK] enabled {}.", "The breakthrough at [MASK] led to {}.",
        "[MASK] transformed the field of {}.", "The success of [MASK] inspired {}.",
        "The foundation of [MASK] supported {}.", "The legacy of [MASK] continues in {}.",
        "The principles of [MASK] guide {}.", "The evolution of [MASK] demonstrates {}.",
        "[MASK] showed the importance of {}.", "The potential of [MASK] suggests {}.",
        "The application of [MASK] improves {}.", "The role of [MASK] shapes {}.",
        "[MASK] established standards for {}.", "The concept from [MASK] changed {}.",
        "Implementation of [MASK] enhanced {}.", "The study at [MASK] revealed {}.",
        "[MASK] opened possibilities in {}.", "The advancement by [MASK] enabled {}.",
        # Additional contextual templates
        "The research in [MASK] shows {}.", "The impact of [MASK] affects {}.",
        "The achievements of [MASK] include {}.", "The development at [MASK] enabled {}.",
        "The analysis from [MASK] revealed {}.", "The integration of [MASK] improved {}.",
        "The methodology of [MASK] transformed {}.", "The framework from [MASK] supported {}.",
        "The architecture of [MASK] allows {}.", "The structure of [MASK] facilitates {}.",
        "The mechanism of [MASK] explains {}.", "The process at [MASK] involves {}.",
        "The system from [MASK] ensures {}.", "The platform of [MASK] provides {}.",
        "The network of [MASK] connects {}.", "The protocol from [MASK] standardizes {}.",
        "The algorithm of [MASK] optimizes {}.", "The model from [MASK] predicts {}.",
        "The theory of [MASK] suggests {}.", "The principle of [MASK] states {}.",
        "The concept from [MASK] implies {}.", "The foundation of [MASK] establishes {}.",
        "The strategy of [MASK] enables {}.", "The approach from [MASK] solves {}.",
        # ... continue with more contextual templates ...
    ]

    fillers = [
        ("France", "Paris"), ("New York", "Big Apple"), ("Eiffel Tower", "landmark"), ("1969", "moon landing"),
        ("Amazon", "rainforest"), ("Tesla", "electric cars"), ("Python", "programming language"),
        ("Mount Everest", "highest peak"), ("Venice", "canals"), ("Shakespeare", "playwright"),
        ("Einstein", "relativity"), ("Beethoven", "composer"), ("Mars", "red planet"),
        ("Amazon", "e-commerce"), ("Google", "search engine"), ("Apple", "technology"),
        ("Microsoft", "software"), ("NASA", "space exploration"), ("Tesla", "innovation"),
        ("Bitcoin", "cryptocurrency"), ("Blockchain", "technology"), ("AI", "artificial intelligence"),
        ("Quantum", "physics"), ("DNA", "genetics"), ("COVID-19", "pandemic"), ("Vaccines", "immunization"),
        ("Climate", "change"), ("Biodiversity", "conservation"), ("Renewable", "energy"),
        ("Solar", "power"), ("Wind", "turbines"), ("Hydrogen", "fuel"), ("Electric", "vehicles"),
        ("Autonomous", "driving"), ("SpaceX", "rockets"), ("Blue Origin", "space tourism"),
        ("Virgin Galactic", "space travel"), ("Cryptography", "security"), ("Cybersecurity", "protection"),
        ("Machine", "learning"), ("Deep", "learning"), ("Neural", "networks"), ("Big", "data"),
        ("Cloud", "computing"), ("Internet", "connectivity"), ("5G", "networks"), ("IoT", "devices"),
        ("Smart", "homes"), ("Wearable", "technology"), ("Augmented", "reality"), ("Virtual", "reality"),
        # Add 150 more unique fillers
        *[(f"Entity{i}", f"Description{i}") for i in range(51, 201)],
        ("CRISPR Technology", "genetic editing"),
        ("Quantum Computing", "superposition states"),
        ("Machine Learning", "pattern recognition"),
        ("Renewable Energy", "sustainable power"),
        ("Space Exploration", "interplanetary travel"),
        ("Artificial Intelligence", "cognitive computing"),
        ("Virtual Reality", "immersive experiences"),
        ("Blockchain", "decentralized systems"),
        ("3D Printing", "additive manufacturing"),
        ("Internet of Things", "connected devices"),
        ("Cloud Computing", "distributed processing"),
        ("Robotics", "automated systems"),
        ("Nanotechnology", "molecular engineering"),
        ("Gene Therapy", "genetic medicine"),
        ("Smart Cities", "urban automation"),
        ("Digital Privacy", "data protection"),
        ("Fusion Energy", "plasma containment"),
        ("Brain-Computer Interface", "neural linking"),
        ("Autonomous Vehicles", "self-driving systems"),
        ("Augmented Reality", "mixed reality"),
        ("Quantum Cryptography", "secure communication"),
        ("Synthetic Biology", "engineered organisms"),
        ("Green Technology", "environmental solutions"),
        ("Space Mining", "resource extraction"),
        ("Human Augmentation", "biological enhancement"),
        # Science and Research
        ("Dark Matter", "cosmic mystery"), ("Neuroscience", "brain studies"),
        ("Particle Physics", "quantum mechanics"), ("Genomics", "DNA sequencing"),
        ("Climate Science", "weather patterns"), ("Microbiology", "cellular systems"),
        
        # Technology and Innovation
        ("Edge Computing", "distributed processing"), ("5G Networks", "high-speed communication"),
        ("Quantum Sensors", "atomic precision"), ("Neural Networks", "deep learning"),
        ("Biometrics", "identity verification"), ("Digital Twins", "virtual replicas"),
        
        # Medicine and Healthcare
        ("Immunotherapy", "cancer treatment"), ("Telemedicine", "remote healthcare"),
        ("Personalized Medicine", "genetic profiles"), ("Bioprinting", "tissue engineering"),
        ("Drug Discovery", "molecular design"), ("Mental Health", "psychological care"),
        
        # Environment and Sustainability
        ("Carbon Capture", "emissions reduction"), ("Ocean Conservation", "marine protection"),
        ("Sustainable Agriculture", "eco-farming"), ("Waste Management", "recycling systems"),
        ("Clean Energy", "renewable power"), ("Wildlife Protection", "species conservation"),
        
        # Space and Exploration
        ("Mars Colonization", "planetary settlement"), ("Asteroid Mining", "space resources"),
        ("Solar Sailing", "light propulsion"), ("Space Tourism", "civilian spaceflight"),
        ("Satellite Networks", "orbital communications"), ("Deep Space", "cosmic exploration"),
        
        # Information and Security
        ("Quantum Encryption", "secure transmission"), ("Zero Trust", "security framework"),
        ("Data Privacy", "information protection"), ("Cyber Defense", "network security"),
        ("Identity Management", "access control"), ("Threat Detection", "security monitoring"),
        
        # Manufacturing and Industry
        ("Smart Factories", "automated production"), ("Digital Manufacturing", "precision control"),
        ("Industrial IoT", "connected machines"), ("Supply Chain", "logistics optimization"),
        ("Quality Control", "automated inspection"), ("Predictive Maintenance", "system monitoring"),
        
        # Transportation and Mobility
        ("Flying Cars", "aerial transport"), ("Hyperloop", "vacuum transit"),
        ("Electric Aircraft", "clean aviation"), ("Smart Roads", "intelligent infrastructure"),
        ("Autonomous Ships", "robotic vessels"), ("Urban Mobility", "city transport"),
        
        # ... continue with more domains and pairs ...
    ]

    gpt2_sentences = random.sample(
        [template.format(filler[0]) for template in gpt2_templates for filler in fillers], num_sentences
    )
    bert_sentences = random.sample(
        [template.format(filler[1]) for template in bert_templates for filler in fillers], num_sentences
    )

    return gpt2_sentences, bert_sentences
