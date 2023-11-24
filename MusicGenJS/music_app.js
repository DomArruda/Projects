async function query(data) {
	const response = await fetch(
		"https://api-inference.huggingface.co/models/facebook/musicgen-stereo-small",
		{
			headers: { Authorization: "Bearer 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" },
			method: "POST",
			body: JSON.stringify(data),
		}
	);
	const result = await response.blob();
	return result;
}

document.getElementById('ml-form').addEventListener('submit', async function(e) {
    e.preventDefault();
    let input = document.getElementById('user-input').value;
    document.getElementById('spinner').style.display = 'block';
    let response = await query({"inputs": input});
    let url = URL.createObjectURL(response);
    let audioPlayer = document.getElementById('audio-player');
    audioPlayer.src = url;
    audioPlayer.style.display = 'block';
    document.getElementById('spinner').style.display = 'none';
    document.getElementById('rating').style.display = 'block';
    document.getElementById('download-link').href = url;
    document.getElementById('download-link').style.display = 'block';
});
