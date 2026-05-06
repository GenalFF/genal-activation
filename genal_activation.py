"""
Genal Activation Function

Author: Genal Lombano
Email: genallombano@gmail.com
Date: May 2026

Definition:
    Genal(x) = x * sigmoid(x / k)
    where k = softplus(theta) + 1e-6
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GenalActivation(nn.Module):
    """
    Genal Activation - A Novel Learnable Activation Function
    
    Genal(x) = x * sigmoid(x / k)
    where k = softplus(theta) + 1e-6
    """
    
    def __init__(self, theta_init=1.0, clamp_range=(-2, 2)):
        super().__init__()
        self.theta = nn.Parameter(torch.tensor(theta_init, dtype=torch.float32))
        self.clamp_range = clamp_range
        self._k_history = []
    
    def forward(self, x):
        # Clamp theta for stability
        theta_clamped = torch.clamp(self.theta, min=self.clamp_range[0], max=self.clamp_range[1])
        
        # Calculate k = softplus(theta) + epsilon
        k = F.softplus(theta_clamped) + 1e-6
        
        # Track k during training
        if self.training:
            self._k_history.append(k.item())
        
        # Genal(x) = x * sigmoid(x / k)
        return x * torch.sigmoid(x / k)
    
    def get_k(self):
        """Returns the current value of k"""
        with torch.no_grad():
            theta_clamped = torch.clamp(self.theta, min=self.clamp_range[0], max=self.clamp_range[1])
            return F.softplus(theta_clamped).item() + 1e-6
    
    def get_k_history(self):
        """Returns the history of k values during training"""
        return self._k_history
    
    def reset_history(self):
        """Resets the k history"""
        self._k_history = []


# Alias for convenience
Genal = GenalActivation


# Example usage when run directly
if __name__ == "__main__":
    print("=" * 50)
    print("Genal Activation - Quick Test")
    print("=" * 50)
    
    # Create activation
    activation = GenalActivation()
    print(f"Initial theta: {activation.theta.item():.4f}")
    print(f"Initial k: {activation.get_k():.4f}")
    
    # Test forward pass
    x = torch.randn(10, 128)
    y = activation(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    print("\n✅ Genal Activation is working!")


# ============================================
# CELDA PARA PROBAR QUE FUNCIONA
# Ejecuta después de crear el archivo
# ============================================

print("\n" + "=" * 50)
print("PROBANDO GENAL ACTIVATION")
print("=" * 50)

# Importar la activación
from genal_activation import GenalActivation
import torch

# Crear instancia
activation = GenalActivation()
print(f"\n📊 Información de la activación:")
print(f"   θ (theta) inicial: {activation.theta.item():.4f}")
print(f"   k inicial: {activation.get_k():.4f}")

# Probar con tensores aleatorios
x = torch.randn(5, 20)
y = activation(x)
print(f"\n📐 Prueba de forma:")
print(f"   Input shape: {x.shape}")
print(f"   Output shape: {y.shape}")

# Probar que es entrenable
optimizer = torch.optim.Adam([activation.theta], lr=0.01)
for i in range(5):
    x = torch.randn(5, 20)
    loss = activation(x).mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

print(f"\n📈 Después de 5 pasos de entrenamiento:")
print(f"   θ (theta) final: {activation.theta.item():.4f}")
print(f"   k final: {activation.get_k():.4f}")

print("\n" + "=" * 50)
print("🎉 TODO FUNCIONA CORRECTAMENTE")
print("   El archivo genal_activation.py ha sido creado")
print("=" * 50)
