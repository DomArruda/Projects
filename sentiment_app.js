async function query(data) {
	const response = await fetch(
		"https://api-inference.huggingface.co/models/nlptown/bert-base-multilingual-uncased-sentiment",
		{
			headers: { Authorization: "PUT YOUR OWN API TOKEN HERE" },
            // https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment?inference_api=true
			method: "POST",
			body: JSON.stringify(data),
		}
	);
	const result = await response.json();
	return result;
}

async function getResult(input) {
    let response = await query({"inputs": input});
    let firstObject = response[0][0];
    let numStars = firstObject['label'];
    let probScore = parseFloat(firstObject['score']).toFixed(2) * 100;
    console.log(`Predicted Score: ${numStars}\nProbability: ${probScore}%`);
    return response;
}

document.getElementById('ml-form').addEventListener('submit', async function(e) {
    e.preventDefault();
    let userInput = document.getElementById('user-input').value;
    let result = await getResult(userInput);
    let resultDiv = document.getElementById('result');
    resultDiv.innerHTML = `<br><strong>\n\nPredicted Score: ${result[0][0].label}<br>Probability: ${(result[0][0].score * 100).toFixed(1)}%</strong>`;;
});
