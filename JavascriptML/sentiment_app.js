async function query(data) {
	const response = await fetch(
		"https://api-inference.huggingface.co/models/nlptown/bert-base-multilingual-uncased-sentiment",
		{
			headers: { Authorization: "Bearer xxxxxxxxxxxxxxxxxxxxxxxxxx" }, // YOU WILL HAVE TO USE YOUR OWN API KEY....
            			method: "POST",
			body: JSON.stringify(data),
		}
	);
	const result = await response.json();
	return result;
}

async function getResult(input) {
    let response = await query({"inputs": input});
    return response;
}

document.getElementById('ml-form').addEventListener('submit', async function(e) {
    e.preventDefault();
    let userInput = document.getElementById('user-input').value;
    let result = await getResult(userInput);
    let resultDiv = document.getElementById('result');
    let html = '<br><strong>Probabilities:</strong><br>';
    let sortedResult = result[0].sort((a, b) => parseInt(a.label) - parseInt(b.label)); // we want it so that 1 star always shows first...
    for (let obj of sortedResult) {
        let numStars = obj['label'];
        let probScore = Math.floor(parseFloat(obj['score']).toFixed(2) * 100);
        html += `${numStars}: ${probScore}%<br>`;
    }
    resultDiv.innerHTML = html;
});
