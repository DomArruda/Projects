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
    // Display the spinner
    document.getElementById('spinner').style.display = 'block';
    let input = document.getElementById('user-input').value;
    let response = await query({"inputs": input});
    let url = URL.createObjectURL(response);
    let audioPlayer = document.getElementById('audio-player');
    audioPlayer.src = url;
    audioPlayer.style.display = 'block';
    // Hide the spinner
    document.getElementById('spinner').style.display = 'none';
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



particlesJS('particles-js',
  
  {
    "particles": {
      "number": {
        "value": 80,
        "density": {
          "enable": true,
          "value_area": 800
        }
      },
      "color": {
        "value": "#ffffff"
      },
      "shape": {
        "type": "circle",
        "stroke": {
          "width": 0,
          "color": "#000000"
        },
        "polygon": {
          "nb_sides": 5
        },
        "image": {
          "src": "img/github.svg",
          "width": 100,
          "height": 100
        }
      },
      "size": {
        "value": 3,  // Adjust the size of the particles
        "random": true,
        "anim": {
          "enable": false,
          "speed": 40,
          "size_min": 0.1,
          "sync": false
        }
      },
      // More options...
    },
    "interactivity": {
      "detect_on": "canvas",
      "events": {
        "onhover": {
          "enable": true,
          "mode": "repulse"
        },
        "onclick": {
          "enable": true,
          "mode": "push"
        },
        "resize": true
      },
      // More options...
    },
    "retina_detect": true
  }

);

