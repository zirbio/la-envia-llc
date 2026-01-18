# Scripts de Configuración

Scripts auxiliares para configurar componentes del sistema de trading.

## setup_twitter_accounts.py

Configura cuentas de Twitter para scraping con `twscrape`.

### Requisitos

Las credenciales deben estar en el archivo `.env`:

```bash
TWITTER_USERNAME=tu_usuario
TWITTER_PASSWORD=tu_password
TWITTER_EMAIL=tu@email.com
TWITTER_EMAIL_PASSWORD=password_email
```

### Uso

```bash
# Ejecutar setup
uv run python scripts/setup_twitter_accounts.py
```

### Qué hace el script

1. ✓ Lee credenciales desde `.env`
2. ✓ Crea pool de cuentas de twscrape
3. ✓ Agrega tu cuenta al pool
4. ✓ Ejecuta login automático
5. ✓ Verifica que la cuenta esté activa

### Salida esperada

```
Adding Twitter account: @Envia_Llc
Logging into Twitter account...
✓ Twitter account configured successfully!
✓ Account @Envia_Llc is ready to use

Total accounts in pool: 1
  - @Envia_Llc: Active

--- Account Verification ---
Username: @Envia_Llc
Email: laenviallc@gmail.com
Active: True
Last login: 2026-01-18 10:30:00
---
```

### Notas

- Las cuentas se guardan en `accounts.db` (archivo local de twscrape)
- Puedes agregar múltiples cuentas para evitar rate limits
- El script es idempotente (puedes ejecutarlo múltiples veces)
- **NUNCA** subas el archivo `accounts.db` a git

### Troubleshooting

**Error: Missing required environment variables**
- Verifica que el archivo `.env` tenga todas las variables de Twitter

**Error al hacer login**
- Twitter podría requerir verificación de email
- Revisa tu email por códigos de verificación
- Asegúrate de que las credenciales sean correctas

**Account: Inactive**
- Ejecuta el script nuevamente para re-login
- Twitter podría haber revocado la sesión
