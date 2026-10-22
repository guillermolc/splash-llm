# Splash Studio

Application macOS nativa (AppKit, sin dependencias) para **configurar, servir y
monitorear** un engine Splash, con panel en vivo estilo MTPLX.

Reemplaza al viejo `splash-menubar`: ahora tiene una ventana con los ajustes del
modelo y un panel "En vivo" que se actualiza cada segundo desde `GET /status`.

## Levantar

```sh
./run.sh                      # compila si es necesario y abre la app
```

Binario: `.build/release/splash-studio`

## La ventana

- **Modelo y ajustes**: modelo (descargados en `install/models/`), puerto,
  contexto (16K–128K/auto), formato KV, liberación inactiva, techo de memoria,
  caché SSD, ANE, prefill int8. Cambiar con el engine encendido pide reinicio.
- **En vivo** (de `/status`): decode (tok/s + ITL p50), prefill (tok/s + TTFT p50),
  draft acceptance, peticiones, scheduler, KV (páginas · GB · contexto),
  caché (hit rate · reusados · disco), memoria (GB / límite · presión),
  térmica, pesos y plan del engine.
- **En vuelo**: una línea por petición activa (fase, prompt, salida, edad).
- Botones **Iniciar / Detener / Liberar memoria**.

En el Dock y en la barra (ícono de cerebro) se mantiene una entrada mínima:
Mostrar ventana / Detener engine / Salir. Al cerrar la ventana la app sigue
viva (vuelve a abrirse desde el Dock).

## Modo CLI (smoke tests)

```sh
.build/release/splash-studio --models
.build/release/splash-studio --status [puerto]
.build/release/splash-studio --release [puerto]
.build/release/splash-studio --logs
```

## Comportamientos

- Si ya hay un engine serviendo en el puerto (p. ej. levantado por otra copia),
  la app lo **adopta** en vez de ignorarlo.
- Al salir, detiene el engine que ella levantó.
- Config en `~/Library/Application Support/splash-menubar/config.json`
  (misma carpeta que el app anterior: tus ajustes se conservan).
- Logs del engine en `~/Library/Application Support/splash-menubar/logs/`.
