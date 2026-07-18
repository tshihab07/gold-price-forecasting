/***** interactive tooltip for trend chart  *****/
(function() {
  'use strict';

  function initChartTooltip(svg, lastPrice) {
    if (!svg) return;

    const tooltip = svg.querySelector('#tooltip');
    if (!tooltip) return;

    const tooltipBg = svg.querySelector('#tooltip-bg');
    const tooltipLabel = svg.querySelector('#tooltip-label');
    const tooltipValue = svg.querySelector('#tooltip-value');
    const tooltipPrice = svg.querySelector('#tooltip-price');
    const triggers = svg.querySelectorAll('.tooltip-trigger');

    if (!triggers.length) return;

    triggers.forEach(trigger => {
      trigger.addEventListener('mouseenter', function(e) {
        const value = parseFloat(this.getAttribute('data-value'));
        const label = this.getAttribute('data-label');
        const isProjected = this.classList.contains('projected') || label.includes('proj');

        tooltipLabel.textContent = label;
        tooltipValue.textContent = 'Return: ' + ((value - lastPrice) / lastPrice * 100).toFixed(2) + '%';
        tooltipPrice.textContent = '$' + value.toFixed(2);

        // position tooltip
        const rect = svg.getBoundingClientRect();
        const svgX = e.clientX - rect.left;
        const svgY = e.clientY - rect.top;

        // keep tooltip within bounds
        const vbWidth = 760;
        const vbHeight = 280;
        let tx = Math.min(Math.max(svgX + 15, 10), vbWidth - 120);
        let ty = Math.min(Math.max(svgY - 30, 10), vbHeight - 60);

        tooltip.setAttribute('transform', `translate(${tx}, ${ty})`);
        tooltip.style.display = 'block';
      });

      trigger.addEventListener('mousemove', function(e) {
        const rect = svg.getBoundingClientRect();
        const svgX = e.clientX - rect.left;
        const svgY = e.clientY - rect.top;

        const vbWidth = 760;
        const vbHeight = 280;
        let tx = Math.min(Math.max(svgX + 15, 10), vbWidth - 120);
        let ty = Math.min(Math.max(svgY - 30, 10), vbHeight - 60);

        tooltip.setAttribute('transform', `translate(${tx}, ${ty})`);
      });

      trigger.addEventListener('mouseleave', function() {
        tooltip.style.display = 'none';
      });
    });
  }

  // Auto-initialize tooltips for all charts
  function autoInitChartTooltips() {
    const charts = document.querySelectorAll('svg[data-last-price]');
    charts.forEach(svg => {
      const lastPrice = parseFloat(svg.getAttribute('data-last-price'));
      if (!isNaN(lastPrice)) {
        initChartTooltip(svg, lastPrice);
      }
    });
  }

  // auto-initialize on DOM ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', autoInitChartTooltips);
  } else {
    autoInitChartTooltips();
  }

  // export for manual initialization if needed
  window.AuricChart = {
    initTooltip: initChartTooltip,
    initAll: autoInitChartTooltips
  };
})();