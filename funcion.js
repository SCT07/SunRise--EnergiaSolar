/* Calculadora de ahorro anual de energía:
Determina cuánto dinero (COP) podría ahorrar un usuario al año si usa paneles solares, sin entrar en detalles técnicos complejos.*/

// El usuario se ahorra un 70% de energía anual($COP).
// Costo: hace referencia al valor mensual que paga el usuario por kW.

//función que calcula el ahorro del 70% en dinero
function AhorrokW(costo, kW) {
  costomensual = costo * kW
  costoanual = costomensual * 12
  descuento = (costoanual * 0.70)
  return descuento;
}

//Se asigna valores a las constantes y se le pasan los valores de acuerdo al id del formulario, el mensaje de error y salida (lugar donde se muestra el resultado)
const formulario = document.getElementById("formularioCalculadora") // Formulario
const mensaje = document.getElementById("mensajeError") // Para mensajes de error
const salida = document.getElementById("salidaDatos") // Para mostrar resultados


// Event listener para el envío del formulario
document.getElementById("formularioCalculadora").addEventListener("submit", function (e) {
  e.preventDefault();

  // Limpia mensajes anteriores y oculta resultados previos
  mensaje.textContent = ""
  salida.style.display = "none"

  //se obtiene  los valores del costo y los kw del formulario
  const costo = parseFloat(document.getElementById("costo_kw").value);
  const kW = parseFloat(document.getElementById("KW").value);

  //Se invoca a la función
  let resultado = AhorrokW(costo, kW)
  console.log(resultado)

  //se Valida el ingreso de solo numeros
  if (isNaN(costo) || isNaN(kW)) {
    salida.textContent = "⚠️ Por favor, ingresa valores válidos.";
    return;
  }

  //se muestra el resultado
  salida.textContent = `Podrías ahorrar aproximadamente $${resultado.toFixed(2)} COP al año usando energía solar.`;
  salida.style.display = "block"
})

//Event listener para el RESET del formulario (LIMPIAR)
formulario.addEventListener("reset", function () {
  // Limpia mensajes de error y salida
  mensaje.textContent = "";
  salida.textContent = "";
  salida.style.display = "none";
});

// estimador de paneles solares necesarios basado en el consumo eléctrico.

//Función: Calculadora de Paneles Solares Necesarios

function calcularPanelesSolares(consumoMensualKWh, horasSolDia, potenciaPanelW) {
  // Conversión de potencia del panel de W a kW
  const potenciaPanelkW = potenciaPanelW / 1000;

  // Energía diaria generada por un panel (kWh/día)
  const energiaPorPanelDia = potenciaPanelkW * horasSolDia;

  // Energía mensual generada por un panel (kWh/mes)
  const energiaPorPanelMes = energiaPorPanelDia * 30; // Aprox. 30 días

  // Número de paneles necesarios (redondeado hacia arriba)
  const panelesNecesarios = Math.ceil(consumoMensualKWh / energiaPorPanelMes);

  return panelesNecesarios;
}

// Seleccionamos elementos del DOM para el segundo formulario
const formularioPaneles = document.getElementById("formularioPaneles");
const mensajePaneles = document.getElementById("mensajeErrorPaneles");
const salidaPaneles = document.getElementById("salidaPaneles");

// Event listener para el envío del formulario de paneles
formularioPaneles.addEventListener("submit", function (e) {
  e.preventDefault();

  // Limpiamos mensajes previos y ocultamos resultados
  mensajePaneles.textContent = "";
  salidaPaneles.style.display = "none";

  // Obtenemos los valores del formulario
  const consumoMensual = parseFloat(document.getElementById("consumoMensual").value);
  const horasSol = parseFloat(document.getElementById("horasSol").value);
  const potenciaPanel = parseFloat(document.getElementById("potenciaPanel").value);

  // Validamos que los valores sean números
  if (isNaN(consumoMensual) || isNaN(horasSol) || isNaN(potenciaPanel)) {
    mensajePaneles.textContent = "⚠️ Por favor, ingresa valores válidos.";
    return;
  }

  // Calculamos los paneles necesarios usando la función
  const panelesNecesarios = calcularPanelesSolares(consumoMensual, horasSol, potenciaPanel);

  // Mostramos el resultado
  salidaPaneles.textContent = `Necesitas aproximadamente ${panelesNecesarios} paneles solares para cubrir tu consumo.`;
  salidaPaneles.style.display = "block";
});

// Event listener para el botón de RESET (limpiar formulario)
formularioPaneles.addEventListener("reset", function () {
  mensajePaneles.textContent = "";
  salidaPaneles.textContent = "";
  salidaPaneles.style.display = "none";
});

