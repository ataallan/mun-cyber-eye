/* Client-side only. Switches a password field between hidden and visible.
   Does not read the value out to the network, storage, or the console. */
(function () {
  function bind(button) {
    var fieldId = button.getAttribute("aria-controls");
    var input = fieldId ? document.getElementById(fieldId) : null;
    if (!input) return;

    var showLabel = button.getAttribute("data-label-show") || "Show password";
    var hideLabel = button.getAttribute("data-label-hide") || "Hide password";
    var showIcon = button.querySelector(".password-toggle-icon-show");
    var hideIcon = button.querySelector(".password-toggle-icon-hide");
    var field = button.closest(".password-field");

    function apply(revealed) {
      input.type = revealed ? "text" : "password";
      button.setAttribute("aria-pressed", revealed ? "true" : "false");
      button.setAttribute("aria-label", revealed ? hideLabel : showLabel);
      if (showIcon) showIcon.hidden = revealed;
      if (hideIcon) hideIcon.hidden = !revealed;
      if (field) field.classList.toggle("is-revealed", revealed);
    }

    apply(false);

    button.addEventListener("click", function (event) {
      event.preventDefault();
      apply(input.type !== "text");
    });
  }

  document.querySelectorAll("[data-password-toggle]").forEach(bind);
})();
