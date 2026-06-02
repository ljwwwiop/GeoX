window.HELP_IMPROVE_VIDEOJS = false;

var INTERP_BASE = "./static/interpolation/stacked";
var NUM_INTERP_FRAMES = 240;

var interp_images = [];
function preloadInterpolationImages() {
  for (var i = 0; i < NUM_INTERP_FRAMES; i++) {
    var path = INTERP_BASE + '/' + String(i).padStart(6, '0') + '.jpg';
    interp_images[i] = new Image();
    interp_images[i].src = path;
  }
}

function setInterpolationImage(i) {
  var image = interp_images[i];
  image.ondragstart = function() { return false; };
  image.oncontextmenu = function() { return false; };
  $('#interpolation-image-wrapper').empty().append(image);
}


$(document).ready(function() {
    // Check for click events on the navbar burger icon
    $(".navbar-burger").click(function() {
      // Toggle the "is-active" class on both the "navbar-burger" and the "navbar-menu"
      $(".navbar-burger").toggleClass("is-active");
      $(".navbar-menu").toggleClass("is-active");

    });

    // Special configuration for method carousel (show 1 image at a time)
    var methodCarouselElement = document.querySelector('#method-carousel');
    if (methodCarouselElement) {
        var methodCarousel = bulmaCarousel.attach('#method-carousel', {
            slidesToScroll: 1,
            slidesToShow: 1,
            loop: true,
            infinite: true,
            autoplay: true,
            autoplaySpeed: 3000,
        });
    }

    // Special configuration for results carousel (show 1 video at a time)
    var resultsCarouselElement = document.querySelector('#results-carousel');
    if (resultsCarouselElement) {
        var resultsCarousel = bulmaCarousel.attach('#results-carousel', {
            slidesToScroll: 1,
            slidesToShow: 1,
            loop: true,
            infinite: true,
            autoplay: true,
            autoplaySpeed: 5000,
        });
    }

    // Special configuration for 3D reconstruction nuScenes carousel
    var reconstructionNuscenesCarouselElement = document.querySelector('#reconstruction-nuscenes-carousel');
    if (reconstructionNuscenesCarouselElement) {
        var reconstructionNuscenesCarousel = bulmaCarousel.attach('#reconstruction-nuscenes-carousel', {
            slidesToScroll: 1,
            slidesToShow: 1,
            loop: true,
            infinite: true,
            autoplay: true,
            autoplaySpeed: 4000,
        });
    }

    // Special configuration for 3D reconstruction DDAD carousel
    var reconstructionDdadCarouselElement = document.querySelector('#reconstruction-ddad-carousel');
    if (reconstructionDdadCarouselElement) {
        var reconstructionDdadCarousel = bulmaCarousel.attach('#reconstruction-ddad-carousel', {
            slidesToScroll: 1,
            slidesToShow: 1,
            loop: true,
            infinite: true,
            autoplay: true,
            autoplaySpeed: 4000,
        });
    }

    // Special configuration for 3D reconstruction Waymo carousel
    var reconstructionWaymoCarouselElement = document.querySelector('#reconstruction-waymo-carousel');
    if (reconstructionWaymoCarouselElement) {
        var reconstructionWaymoCarousel = bulmaCarousel.attach('#reconstruction-waymo-carousel', {
            slidesToScroll: 1,
            slidesToShow: 1,
            loop: true,
            infinite: true,
            autoplay: true,
            autoplaySpeed: 4000,
        });
    }

    // Special configuration for 3D reconstruction OpenScene carousel
    var reconstructionOpensceneCarouselElement = document.querySelector('#reconstruction-openscene-carousel');
    if (reconstructionOpensceneCarouselElement) {
        var reconstructionOpensceneCarousel = bulmaCarousel.attach('#reconstruction-openscene-carousel', {
            slidesToScroll: 1,
            slidesToShow: 1,
            loop: true,
            infinite: true,
            autoplay: true,
            autoplaySpeed: 4000,
        });
    }

    // Special configuration for 3D reconstruction PandaSet carousel
    var reconstructionPandasetCarouselElement = document.querySelector('#reconstruction-pandaset-carousel');
    if (reconstructionPandasetCarouselElement) {
        var reconstructionPandasetCarousel = bulmaCarousel.attach('#reconstruction-pandaset-carousel', {
            slidesToScroll: 1,
            slidesToShow: 1,
            loop: true,
            infinite: true,
            autoplay: true,
            autoplaySpeed: 4000,
        });
    }

    // Special configuration for 3D reconstruction Lyft carousel
    var reconstructionLyftCarouselElement = document.querySelector('#reconstruction-lyft-carousel');
    if (reconstructionLyftCarouselElement) {
        var reconstructionLyftCarousel = bulmaCarousel.attach('#reconstruction-lyft-carousel', {
            slidesToScroll: 1,
            slidesToShow: 1,
            loop: true,
            infinite: true,
            autoplay: true,
            autoplaySpeed: 4000,
        });
    }

    // Special configuration for 3D reconstruction AV2 carousel
    var reconstructionAv2CarouselElement = document.querySelector('#reconstruction-av2-carousel');
    if (reconstructionAv2CarouselElement) {
        var reconstructionAv2Carousel = bulmaCarousel.attach('#reconstruction-av2-carousel', {
            slidesToScroll: 1,
            slidesToShow: 1,
            loop: true,
            infinite: true,
            autoplay: true,
            autoplaySpeed: 4000,
        });
    }

    var options = {
        slidesToScroll: 1,
        slidesToShow: 1,
        loop: true,
        infinite: true,
        autoplay: false,
        autoplaySpeed: 3000,
    }

    // Initialize all remaining div with carousel class (if any)
    var carousels = bulmaCarousel.attach('.carousel', options);

    // Loop on each carousel initialized
    for(var i = 0; i < carousels.length; i++) {
    	// Add listener to  event
    	carousels[i].on('before:show', state => {
    		console.log(state);
    	});
    }

    // Access to bulmaCarousel instance of an element
    var element = document.querySelector('#my-element');
    if (element && element.bulmaCarousel) {
    	// bulmaCarousel instance is available as element.bulmaCarousel
    	element.bulmaCarousel.on('before-show', function(state) {
    		console.log(state);
    	});
    }

    /*var player = document.getElementById('interpolation-video');
    player.addEventListener('loadedmetadata', function() {
      $('#interpolation-slider').on('input', function(event) {
        console.log(this.value, player.duration);
        player.currentTime = player.duration / 100 * this.value;
      })
    }, false);*/
    preloadInterpolationImages();

    $('#interpolation-slider').on('input', function(event) {
      setInterpolationImage(this.value);
    });
    setInterpolationImage(0);
    $('#interpolation-slider').prop('max', NUM_INTERP_FRAMES - 1);

    bulmaSlider.attach();

})
