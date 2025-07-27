"use client"

import { Smartphone, Shirt, Home, Book, Gamepad2, Watch, Headphones, Camera } from "lucide-react"

interface WelcomeSectionProps {
  onCategoryClick: (query: string) => void
}

const categories = [
  { name: "Mobiles", icon: Smartphone, query: "mobile phone", color: "bg-blue-100" },
  { name: "Fashion", icon: Shirt, query: "fashion clothes", color: "bg-pink-100" },
  { name: "Electronics", icon: Camera, query: "electronics", color: "bg-purple-100" },
  { name: "Home & Kitchen", icon: Home, query: "home kitchen", color: "bg-green-100" },
  { name: "Books", icon: Book, query: "books", color: "bg-yellow-100" },
  { name: "Sports", icon: Gamepad2, query: "sports", color: "bg-red-100" },
  { name: "Watches", icon: Watch, query: "watches", color: "bg-indigo-100" },
  { name: "Audio", icon: Headphones, query: "headphones speakers", color: "bg-gray-100" },
]

const featuredProducts = [
  {
    title: "Best of Electronics",
    subtitle: "Top Deals on Gadgets",
    image: "/placeholder.svg?height=200&width=300",
    query: "electronics",
  },
  {
    title: "Fashion Trends",
    subtitle: "Latest Styles for You",
    image: "/placeholder.svg?height=200&width=300",
    query: "fashion",
  },
  {
    title: "Home Essentials",
    subtitle: "Everything for Your Home",
    image: "/placeholder.svg?height=200&width=300",
    query: "home essentials",
  },
]

export default function WelcomeSection({ onCategoryClick }: WelcomeSectionProps) {
  return (
    <div className="px-4 py-4 sm:py-8">
      {/* Welcome Message */}
      <div className="text-center mb-6 sm:mb-8">
        <h1 className="text-2xl sm:text-3xl font-bold text-gray-800 mb-2">Welcome to Flipkart</h1>
        <p className="text-sm sm:text-base text-gray-600">Discover millions of products at the best prices</p>
      </div>

      {/* Category Grid */}
      <div className="mb-8 sm:mb-12">
        <h2 className="text-lg sm:text-xl font-semibold mb-4 sm:mb-6">Shop by Category</h2>
        <div className="grid grid-cols-4 sm:grid-cols-4 md:grid-cols-6 lg:grid-cols-8 gap-2 sm:gap-4">
          {categories.map((category) => {
            const IconComponent = category.icon
            return (
              <div
                key={category.name}
                onClick={() => onCategoryClick(category.query)}
                className="flex flex-col items-center p-2 sm:p-4 rounded-lg cursor-pointer hover:shadow-md transition-shadow bg-white border"
              >
                <div className={`p-2 sm:p-3 rounded-full ${category.color} mb-1 sm:mb-2`}>
                  <IconComponent size={16} className="sm:w-6 sm:h-6 text-gray-700" />
                </div>
                <span className="text-xs sm:text-sm font-medium text-center">{category.name}</span>
              </div>
            )
          })}
        </div>
      </div>

      {/* Featured Sections */}
      <div className="mb-6 sm:mb-8">
        <h2 className="text-lg sm:text-xl font-semibold mb-4 sm:mb-6">Featured Collections</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4 sm:gap-6">
          {featuredProducts.map((product, index) => (
            <div
              key={index}
              onClick={() => onCategoryClick(product.query)}
              className="bg-white rounded-lg shadow-sm hover:shadow-md transition-shadow cursor-pointer overflow-hidden"
            >
              <img
                src={product.image || "/placeholder.svg"}
                alt={product.title}
                className="w-full h-32 sm:h-48 object-cover"
              />
              <div className="p-3 sm:p-4">
                <h3 className="font-semibold text-base sm:text-lg mb-1">{product.title}</h3>
                <p className="text-gray-600 text-xs sm:text-sm">{product.subtitle}</p>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Popular Searches */}
      <div>
        <h2 className="text-lg sm:text-xl font-semibold mb-3 sm:mb-4">Popular Searches</h2>
        <div className="flex flex-wrap gap-2">
          {[
            "iPhone 15",
            "Samsung Galaxy",
            "Laptop",
            "Headphones",
            "Shoes",
            "Watch",
            "Kurta",
            "Jeans",
            "Books",
            "Mobile Cover",
          ].map((search) => (
            <button
              key={search}
              onClick={() => onCategoryClick(search)}
              className="bg-white border border-gray-200 hover:border-blue-300 px-2 sm:px-4 py-1 sm:py-2 rounded-full text-xs sm:text-sm transition-colors"
            >
              {search}
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}
