import re
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import spacy
except ImportError:
    spacy = None

try:
    from gensim.models import Word2Vec
except ImportError:
    Word2Vec = None


class MobileRecommender:
    def __init__(self, csv_path):
        self.csv_path = Path(csv_path)
        self.df = self._load_dataset()
        self.nlp = self._load_spacy()

        # Each phone is converted into one searchable text document.
        self.documents = self.df.apply(self._make_description, axis=1).tolist()
        self.tokenized_documents = [self.preprocess_text(text) for text in self.documents]

        # Word2Vec learns semantic meaning from the dataset itself.
        self.word2vec_model = self._train_word2vec()
        self.phone_vectors = np.array(
            [self._document_vector(tokens) for tokens in self.tokenized_documents]
        )

    def _load_dataset(self):
        df = pd.read_csv(self.csv_path, encoding="utf-8")
        df = df.fillna("")
        df["brand"] = df["model"].apply(lambda value: str(value).split()[0])
        df["numeric_price"] = df["price"].apply(self._extract_number)
        df["numeric_rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0)
        return df

    def _load_spacy(self):
        if spacy is None:
            return None
        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            return spacy.blank("en")

    def _make_description(self, row):
        return " ".join(
            [
                str(row["model"]),
                str(row["brand"]),
                str(row["price"]),
                str(row["rating"]),
                str(row["sim"]),
                str(row["processor"]),
                str(row["ram"]),
                str(row["battery"]),
                str(row["display"]),
                str(row["camera"]),
                str(row["card"]),
                str(row["os"]),
                self._feature_keywords(row),
            ]
        )

    def _feature_keywords(self, row):
        text = " ".join(str(row.get(col, "")) for col in self.df.columns).lower()
        keywords = []
        if "5g" in text:
            keywords.extend(["5g", "fast internet", "modern network"])
        if any(word in text for word in ["snapdragon 8", "dimensity 900", "a15", "a16"]):
            keywords.extend(["gaming", "performance", "powerful processor"])
        if "5000" in text or "6000" in text:
            keywords.extend(["strong battery", "long usage", "battery backup"])
        if any(word in text for word in ["64", "108", "200", "triple rear"]):
            keywords.extend(["camera", "photography", "photos"])
        return " ".join(keywords)

    def preprocess_text(self, text):
        text = str(text).lower()
        text = text.replace("â‚¹", " rupees ")
        text = text.replace("â€‰", " ")
        text = re.sub(r"[^a-z0-9\s]", " ", text)

        if self.nlp is None:
            return [
                token
                for token in text.split()
                if token not in {"the", "a", "an", "with", "and", "for", "of", "to"}
            ]

        doc = self.nlp(text)
        tokens = []
        for token in doc:
            if token.is_space or token.is_punct or token.is_stop:
                continue
            lemma = token.lemma_.strip() if token.lemma_ else token.text
            if lemma:
                tokens.append(lemma.lower())
        return tokens

    def _train_word2vec(self):
        if Word2Vec is None:
            return None
        return Word2Vec(
            sentences=self.tokenized_documents,
            vector_size=100,
            window=5,
            min_count=1,
            workers=1,
            sg=1,
            epochs=80,
            seed=42,
        )

    def _document_vector(self, tokens):
        if not tokens:
            return np.zeros(100)

        if self.word2vec_model is None:
            return self._simple_vector(tokens)

        vectors = [
            self.word2vec_model.wv[token]
            for token in tokens
            if token in self.word2vec_model.wv
        ]
        if not vectors:
            return np.zeros(self.word2vec_model.vector_size)
        return np.mean(vectors, axis=0)

    def _simple_vector(self, tokens):
        # Fallback keeps the app usable before installing spaCy and Gensim.
        vector = np.zeros(100)
        for token in tokens:
            index = abs(hash(token)) % 100
            vector[index] += 1
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector

    def recommend(self, query, top_n=5):
        query_tokens = self.preprocess_text(query)
        query_vector = self._document_vector(query_tokens)
        budget = self._extract_budget(query)
        requested_brand = self._extract_brand(query)

        scores = []
        for index, phone_vector in enumerate(self.phone_vectors):
            # Cosine similarity compares the meaning of the query and phone specs.
            similarity = self._cosine_similarity(query_vector, phone_vector)
            row = self.df.iloc[index]

            # Small boosts make explicit budget and brand requests more practical.
            if budget and row["numeric_price"] <= budget:
                similarity += 0.08
            if requested_brand and row["brand"].lower() == requested_brand:
                similarity += 0.08

            scores.append((index, min(similarity, 1.0)))

        scores.sort(key=lambda item: item[1], reverse=True)
        recommendations = []
        for rank, (index, score) in enumerate(scores[:top_n], start=1):
            phone = self._row_to_phone(self.df.iloc[index])
            phone["score"] = round(score * 100, 2)
            phone["rank"] = rank
            phone["explanation"] = self._build_explanation(query, phone, budget)
            recommendations.append(phone)
        return recommendations

    def _cosine_similarity(self, vector_one, vector_two):
        denominator = np.linalg.norm(vector_one) * np.linalg.norm(vector_two)
        if denominator == 0:
            return 0
        return float(np.dot(vector_one, vector_two) / denominator)

    def _extract_number(self, value):
        numbers = re.findall(r"\d+", str(value).replace(",", ""))
        return int("".join(numbers)) if numbers else 0

    def _extract_budget(self, query):
        query = query.lower().replace(",", "")
        match = re.search(r"(?:under|below|less than|within)\s*(?:rs|₹|rupees)?\s*(\d+)", query)
        if match:
            return int(match.group(1))
        match = re.search(r"(\d+)\s*k", query)
        if match:
            return int(match.group(1)) * 1000
        return None

    def _extract_brand(self, query):
        brands = {brand.lower() for brand in self.df["brand"].unique()}
        tokens = re.findall(r"[a-zA-Z]+", query.lower())
        for token in tokens:
            if token in brands:
                return token
        return None

    def _build_explanation(self, query, phone, budget):
        query_lower = query.lower()
        combined_specs = " ".join(
            [
                phone["model"],
                phone["sim"],
                phone["processor"],
                phone["ram"],
                phone["battery"],
                phone["camera"],
                str(phone["rating"]),
            ]
        ).lower()
        reasons = []

        if budget and phone["numeric_price"] <= budget:
            reasons.append(f"Fits your Rs. {budget:,} budget at {phone['price']}")

        if phone["brand"].lower() in query_lower:
            reasons.append(f"Matches your preferred {phone['brand']} brand")

        if "gaming" in query_lower or "performance" in query_lower:
            reasons.append(f"Uses {self._short_processor(phone['processor'])} for better performance")

        if "battery" in query_lower or "long" in query_lower:
            battery = self._extract_battery_capacity(phone["battery"])
            reasons.append(
                f"{battery} battery is useful for long usage"
                if battery
                else f"Battery details match your backup requirement: {phone['battery']}"
            )

        if "camera" in query_lower or "photo" in query_lower or "photography" in query_lower:
            reasons.append(f"Camera setup suits photography: {self._short_camera(phone['camera'])}")

        if "5g" in query_lower and "5g" in combined_specs:
            reasons.append("Supports 5G connectivity")

        if "storage" in query_lower or "space" in query_lower or "memory" in query_lower:
            reasons.append(f"Offers {self._short_ram_storage(phone['ram'])}")

        if len(reasons) < 3:
            reasons.extend(self._general_phone_strengths(phone))

        if not reasons:
            reasons.append("Its specifications are semantically close to your search")

        return self._unique_reasons(reasons)[:4]

    def _short_processor(self, processor):
        processor = str(processor)
        return processor.split(",")[0].strip() or "a capable processor"

    def _short_camera(self, camera):
        camera = str(camera)
        return camera.split("&")[0].strip() or "rear camera setup"

    def _short_ram_storage(self, ram):
        return str(ram).replace("\u2009", " ").strip()

    def _extract_battery_capacity(self, battery):
        match = re.search(r"(\d{4,5})\s*mAh", str(battery), re.IGNORECASE)
        return f"{match.group(1)} mAh" if match else ""

    def _general_phone_strengths(self, phone):
        strengths = []
        processor = self._short_processor(phone["processor"])
        battery = self._extract_battery_capacity(phone["battery"])
        rating = float(phone["rating"]) if str(phone["rating"]).replace(".", "", 1).isdigit() else 0

        if processor:
            strengths.append(f"{processor} gives this model a clear performance advantage")
        if battery:
            strengths.append(f"{battery} battery makes it practical for daily use")
        if "5g" in phone["sim"].lower():
            strengths.append("5G support makes it ready for faster networks")
        if rating >= 80:
            strengths.append(f"Rating of {phone['rating']} shows strong overall value")
        elif phone["rating"]:
            strengths.append(f"Rating of {phone['rating']} adds confidence for this choice")
        strengths.append(f"{self._short_ram_storage(phone['ram'])} is a good memory and storage combination")
        return strengths

    def _unique_reasons(self, reasons):
        unique = []
        seen = set()
        for reason in reasons:
            key = reason.lower()
            if key not in seen:
                unique.append(reason)
                seen.add(key)
        return unique

    def _row_to_phone(self, row):
        return {
            "model": row["model"],
            "brand": row["brand"],
            "price": row["price"],
            "numeric_price": row["numeric_price"],
            "rating": row["rating"],
            "sim": row["sim"],
            "processor": row["processor"],
            "ram": row["ram"],
            "battery": row["battery"],
            "display": row["display"],
            "camera": row["camera"],
            "storage": self._extract_storage(row["ram"]),
            "card": row["card"],
            "os": row["os"],
        }

    def _extract_storage(self, value):
        match = re.search(r"(\d+\s*GB|\d+\s*TB)\s*inbuilt", str(value), re.IGNORECASE)
        return match.group(1) if match else "See RAM/Storage details"

    def get_phone_options(self):
        return self.df["model"].sort_values().tolist()

    def get_phone_by_model(self, model):
        match = self.df[self.df["model"] == model]
        if match.empty:
            return None
        return self._row_to_phone(match.iloc[0])
