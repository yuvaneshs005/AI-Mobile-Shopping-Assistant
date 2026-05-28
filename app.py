from flask import Flask, render_template, request

from recommender import MobileRecommender


app = Flask(__name__)
recommender = MobileRecommender("data/smartphones.csv")


@app.route("/")
def home():
    examples = [
        "Samsung phone under 25000 with good camera",
        "Gaming phone with strong battery",
        "Best phone for photography",
        "5G phone with good performance",
    ]
    return render_template("index.html", examples=examples)


@app.route("/recommend", methods=["POST"])
def recommend():
    query = request.form.get("query", "").strip()
    if not query:
        return render_template(
            "results.html",
            query=query,
            recommendations=[],
            message="Please enter what kind of phone you are looking for.",
        )

    recommendations = recommender.recommend(query, top_n=5)
    return render_template(
        "results.html",
        query=query,
        recommendations=recommendations,
        message=None,
    )


@app.route("/compare", methods=["GET", "POST"])
def compare():
    phones = recommender.get_phone_options()
    selected_one = request.form.get("phone_one", "") if request.method == "POST" else ""
    selected_two = request.form.get("phone_two", "") if request.method == "POST" else ""
    phone_one = recommender.get_phone_by_model(selected_one) if selected_one else None
    phone_two = recommender.get_phone_by_model(selected_two) if selected_two else None

    return render_template(
        "compare.html",
        phones=phones,
        selected_one=selected_one,
        selected_two=selected_two,
        phone_one=phone_one,
        phone_two=phone_two,
    )


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
