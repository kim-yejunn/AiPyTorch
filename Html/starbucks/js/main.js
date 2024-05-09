//search
const searchEl = document.querySelector('.search');
//input
const searchInputEl = searchEl.querySelector('input');

searchEl.addEventListener('clik',function (){
  searchInputEl.focus();
});
searchInputEl.addEventListener('focus',function (){
  searchEl.classList.add('focused');
  searchInputEl.setAttribute('placeholder','통합검색');
});
searchInputEl.addEventListener('blur',function (){
  searchEl.classList.remove('focused');
  searchInputEl.setAttribute('placeholder','');
});
const badgeEl = document.querySelector('header .badges');
const toTopEl = document.querySelector('#to-top');
window.addEventListener('scroll', _.throttle(function(){
  if(window.scrollY > 500){
    // badgeEl.style.diplay='none';
    // gsap.to(요소, 시간, 옵션);
    gsap.to(badgeEl, 0.6,{
      opacity : 0,
      display : 'none'
    });
    //to-top보이게
    //to-top 숨기기
    gsap.to(toTopEl, .2, {
      x:0
    });
    }else{
    // badgeEl.style.diplay='block';
    gsap.to(badgeEl, 0.6,{
      opacity : 1,
      display : 'block'
    });
    //to-top 숨기기
    gsap.to(toTopEl, .2, {
      x:100
    });
  }
}, 300));
//클릭하면 top이동
toTopEl.addEventListener('click',function(){
  gsap.to(window, .7,{
    scrollTo: 0
  });
});

//visual 서서히 나타나기
const fadeEls = document.querySelectorAll('.visual .fade-in');
fadeEls.forEach(function(fadeEl, index){
gsap.to(fadeEl, 1,{
  delay:(index + 1) * 0.7,
  opacity:1
});
});


//자동으로 올해 년도 찾기
const thisYear = document.querySelector('.this-year');
thisYear.textContent = new Date().getFullYear()

new Swiper('.promotion .swiper-container',{
  autoplay:{
    delay:5000 //5초마다 슬라이드 변경
  },
slidesPerView: 3, //화면에 몇개 보일것인지
spaceBetween: 10, //슬라이더 사이 간격
centeredSlides: true, //1번 슬라이더 센터
pagination:{
  el: '.promotion .swiper-pagination',
  clickable: true
},
navigation:{
prevEl:'.promotion .swiper-prev',
nextEl:'.promotion .swiper-next'
}
})