from flask import Flask, jsonify, request
# other imports as well
app = Flask(__name__)

# Endpoint to serve survey questions
@app.route('/survey', methods=['GET'])
def serve_survey():
    questions = [
        {
            "question": "Would you like to filter out songs with explicit content?",
            "options": ["Y", "N"],
            "type": "yes_no"
        },
        {
            "question": "There's a lot of different genres! Would you like to see only the most popular ones?",
            "options": ["Y", "N"],
            "type": "yes_no"
        },
        {
            "question": ("Would you like to specify how much of each audio feature you would like? "
                         "(e.g. loudness, tempo, etc). It's not required, but it is highly recommended for a better recommendation!"),
            "options": ["Y", "N"],
            "type": "yes_no"
        },
        {
            "question": "What's your top five favorite songs?",
            "options": [],
            "type": "text"
        },
        {
            "question": "Pick a genre from the list!",
            "options": [],  # We'll need to fetch genres dynamically, but for now it's empty.
            "type": "text"
        }
    ]
    return jsonify(questions)


def get_music_recommendations(user_answers):
    recommendations = []
    if user_answers.get("Would you like to filter out songs with explicit content?") == 'Y':
        recommendations.extend([" Song 1", " Song 2", " Song 3", "......."])
    else:
        recommendations.extend(["ex Song 4", "ex Song 5", "ex Song 6"])
    return recommendations

@app.route('/recommendations', methods=['POST'])
def generate_recommendations():
    answers = request.json
    recommendations = get_music_recommendations(answers)
    return jsonify(recommendations)
