"use client"

import { useState, useEffect } from "react"
import { X, Plus, Minus, Trash2 } from "lucide-react"

interface CartModalProps {
  onClose: () => void
}

interface CartItem {
  id: number
  title: string
  price: number
  quantity: number
  thumbnail?: string
}

export default function CartModal({ onClose }: CartModalProps) {
  const [cartItems, setCartItems] = useState<CartItem[]>([])

  useEffect(() => {
    const cart = JSON.parse(localStorage.getItem("cart") || "[]")
    setCartItems(cart)
  }, [])

  const updateQuantity = (id: number, newQuantity: number) => {
    if (newQuantity === 0) {
      removeItem(id)
      return
    }

    const updatedCart = cartItems.map((item) => (item.id === id ? { ...item, quantity: newQuantity } : item))
    setCartItems(updatedCart)
    localStorage.setItem("cart", JSON.stringify(updatedCart))

    // Trigger storage event for cart count update
    window.dispatchEvent(
      new StorageEvent("storage", {
        key: "cart",
        newValue: JSON.stringify(updatedCart),
      }),
    )
  }

  const removeItem = (id: number) => {
    const updatedCart = cartItems.filter((item) => item.id !== id)
    setCartItems(updatedCart)
    localStorage.setItem("cart", JSON.stringify(updatedCart))

    // Trigger storage event for cart count update
    window.dispatchEvent(
      new StorageEvent("storage", {
        key: "cart",
        newValue: JSON.stringify(updatedCart),
      }),
    )
  }

  const getTotalPrice = () => {
    return cartItems.reduce((total, item) => total + item.price * item.quantity, 0)
  }

  const getTotalItems = () => {
    return cartItems.reduce((total, item) => total + item.quantity, 0)
  }

  const handleCheckout = () => {
    const currentUser = JSON.parse(localStorage.getItem("currentUser") || "null")

    if (!currentUser) {
      alert("Please login to proceed with checkout")
      return
    }

    if (cartItems.length === 0) {
      alert("Your cart is empty!")
      return
    }

    // Simple checkout simulation
    const order = {
      id: Date.now(),
      userId: currentUser.id,
      items: cartItems,
      total: getTotalPrice(),
      status: "confirmed",
      createdAt: new Date().toISOString(),
    }

    const orders = JSON.parse(localStorage.getItem("orders") || "[]")
    orders.push(order)
    localStorage.setItem("orders", JSON.stringify(orders))

    // Clear cart
    setCartItems([])
    localStorage.setItem("cart", JSON.stringify([]))

    // Trigger storage event
    window.dispatchEvent(
      new StorageEvent("storage", {
        key: "cart",
        newValue: JSON.stringify([]),
      }),
    )

    alert(`Order placed successfully! Order ID: ${order.id}`)
    onClose()
  }

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-2 sm:p-4">
      <div className="bg-white rounded-lg w-full max-w-2xl mx-2 sm:mx-4 max-h-[90vh] sm:max-h-[80vh] flex flex-col">
        <div className="flex justify-between items-center p-4 border-b">
          <h2 className="text-xl font-semibold">My Cart ({getTotalItems()} items)</h2>
          <button onClick={onClose} className="text-gray-500 hover:text-gray-700">
            <X size={24} />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-4">
          {cartItems.length === 0 ? (
            <div className="text-center py-8">
              <div className="text-gray-400 mb-4">
                <svg className="w-16 h-16 mx-auto" fill="currentColor" viewBox="0 0 20 20">
                  <path
                    fillRule="evenodd"
                    d="M10 2L3 7v11a2 2 0 002 2h10a2 2 0 002-2V7l-7-5zM8 16a1 1 0 100-2 1 1 0 000 2zm4 0a1 1 0 100-2 1 1 0 000 2z"
                    clipRule="evenodd"
                  />
                </svg>
              </div>
              <p className="text-gray-600 mb-4">Your cart is empty</p>
              <button
                onClick={onClose}
                className="bg-[#2874f0] text-white px-6 py-2 rounded-md hover:bg-blue-600 transition-colors"
              >
                Continue Shopping
              </button>
            </div>
          ) : (
            <div className="space-y-4">
              {cartItems.map((item) => (
                <div key={item.id} className="flex items-center space-x-2 sm:space-x-4 p-2 sm:p-4 border rounded-lg">
                  <img
                    src={item.thumbnail || "/placeholder.svg?height=80&width=80"}
                    alt={item.title}
                    className="w-12 h-12 sm:w-16 sm:h-16 object-cover rounded"
                  />

                  <div className="flex-1 min-w-0">
                    <h3 className="font-medium text-xs sm:text-sm mb-1 line-clamp-2">{item.title}</h3>
                    <p className="text-sm sm:text-lg font-semibold">₹{item.price.toLocaleString()}</p>
                  </div>

                  <div className="flex items-center space-x-1 sm:space-x-2">
                    <button
                      onClick={() => updateQuantity(item.id, item.quantity - 1)}
                      className="p-1 border rounded hover:bg-gray-100"
                    >
                      <Minus size={12} className="sm:w-4 sm:h-4" />
                    </button>
                    <span className="w-6 sm:w-8 text-center text-sm">{item.quantity}</span>
                    <button
                      onClick={() => updateQuantity(item.id, item.quantity + 1)}
                      className="p-1 border rounded hover:bg-gray-100"
                    >
                      <Plus size={12} className="sm:w-4 sm:h-4" />
                    </button>
                  </div>

                  <button
                    onClick={() => removeItem(item.id)}
                    className="p-1 sm:p-2 text-red-500 hover:bg-red-50 rounded"
                  >
                    <Trash2 size={14} className="sm:w-4 sm:h-4" />
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>

        {cartItems.length > 0 && (
          <div className="border-t p-4">
            <div className="flex justify-between items-center mb-4">
              <span className="text-lg font-semibold">Total: ₹{getTotalPrice().toLocaleString()}</span>
              <span className="text-sm text-gray-600">({getTotalItems()} items)</span>
            </div>

            <div className="flex flex-col sm:flex-row space-y-2 sm:space-y-0 sm:space-x-3">
              <button
                onClick={onClose}
                className="flex-1 border border-gray-300 text-gray-700 py-2 sm:py-3 px-4 rounded-md hover:bg-gray-50 transition-colors text-sm sm:text-base"
              >
                Continue Shopping
              </button>
              <button
                onClick={handleCheckout}
                className="flex-1 bg-[#fb641b] text-white py-2 sm:py-3 px-4 rounded-md hover:bg-orange-600 transition-colors font-medium text-sm sm:text-base"
              >
                Checkout
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
