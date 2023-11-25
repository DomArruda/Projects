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
    let response = await query({"inputs": input});
    let url = URL.createObjectURL(response);
    let audioPlayer = document.getElementById('audio-player');
    audioPlayer.src = url;
    audioPlayer.style.display = 'block';
    // Display the star rating section and review text
    document.querySelector('.ratings').style.display = 'flex';
    document.querySelector('.review-text').style.display = 'block';
});

let stars = document.querySelectorAll(".ratings span");

for(let star of stars){
   star.addEventListener("click", function(){
      
      let children = 	star.parentElement.children;
      for(let child of children){
         if(child.getAttribute("data-clicked")){
            child.removeAttribute("data-clicked");
            return false;	
         }
      }
      
      this.setAttribute("data-clicked","true");
   });
}
