"use client"

import { useState } from "react"

interface Filters {
  categories: string[]
  brands: string[]
  rating: number
  priceMin: number
  priceMax: number
  deliveryDays: number
  deliveryFilterEnabled: boolean
}

interface FiltersSidebarProps {
  filters: Filters
  onChange: (filters: Filters) => void
  isLoading?: boolean
}

const categories = ["Electronics", "Fashion", "Home & Kitchen", "Books", "Sports"]

const brands = ["Apple", "Samsung", "Nike", "Adidas", "Sony", "LG", "Boat", "Realme"]

export default function FiltersSidebar({ filters, onChange, isLoading = false }: FiltersSidebarProps) {
  const [isDeliveryChanging, setIsDeliveryChanging] = useState(false)

  const handleCategoryChange = (category: string, checked: boolean) => {
    const newCategories = checked ? [...filters.categories, category] : filters.categories.filter((c) => c !== category)
    onChange({ ...filters, categories: newCategories })
  }

  const handleBrandChange = (brand: string, checked: boolean) => {
    const newBrands = checked ? [...filters.brands, brand] : filters.brands.filter((b) => b !== brand)
    onChange({ ...filters, brands: newBrands })
  }

  const handleDeliveryChange = (value: number) => {
    setIsDeliveryChanging(true)
    onChange({ ...filters, deliveryDays: value })
    // Reset the changing state after a short delay
    setTimeout(() => setIsDeliveryChanging(false), 300)
  }

  return (
    <div className="w-full lg:w-64 bg-white border-r lg:min-h-screen p-4">
      <h3 className="font-semibold text-lg mb-4 lg:block hidden">Filters</h3>

      {/* Mobile: Show filters in grid, Desktop: Show in column */}
      <div className="space-y-4 lg:space-y-6">
        {/* Delivery Days - First Priority */}
        <div className="lg:mb-6">
          <h4 className="font-medium mb-3 text-sm lg:text-base flex items-center justify-between">
            <span className="flex items-center">
              🚚 Delivery 
              {(isLoading || isDeliveryChanging) && (
                <div className="ml-2 w-3 h-3 border-2 border-blue-600 border-t-transparent rounded-full animate-spin"></div>
              )}
            </span>
            <label className="relative inline-flex items-center cursor-pointer">
              <input
                type="checkbox"
                checked={filters.deliveryFilterEnabled}
                onChange={(e) => onChange({ ...filters, deliveryFilterEnabled: e.target.checked })}
                className="sr-only peer"
              />
              <div className="w-9 h-5 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-blue-600"></div>
            </label>
          </h4>
          
          {filters.deliveryFilterEnabled && (
            <div className="mb-2">
              <label className={`text-xs lg:text-sm transition-all duration-300 ${isDeliveryChanging ? 'text-blue-600 font-medium' : 'text-gray-700'}`}>
                Within {filters.deliveryDays} day{filters.deliveryDays > 1 ? 's' : ''}
              </label>
              <div className="relative">
                <input
                  type="range"
                  min="1"
                  max="7"
                  value={filters.deliveryDays}
                  onChange={(e) => handleDeliveryChange(Number(e.target.value))}
                  className="w-full mt-1 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer slider-thumb transition-all duration-200 hover:bg-gray-300"
                  style={{
                    background: `linear-gradient(to right, #3b82f6 0%, #3b82f6 ${((filters.deliveryDays - 1) / 6) * 100}%, #e5e7eb ${((filters.deliveryDays - 1) / 6) * 100}%, #e5e7eb 100%)`
                  }}
                />
                {isDeliveryChanging && (
                  <div className="absolute -top-8 left-1/2 transform -translate-x-1/2 bg-blue-600 text-white text-xs px-2 py-1 rounded animate-pulse">
                    {filters.deliveryDays} day{filters.deliveryDays > 1 ? 's' : ''}
                  </div>
                )}
              </div>
              <div className="flex justify-between text-xs text-gray-500 mt-1">
                <span className="transition-colors duration-200 hover:text-gray-700">1 day</span>
                <span className="transition-colors duration-200 hover:text-gray-700">7 days</span>
              </div>
            </div>
          )}
          
          {!filters.deliveryFilterEnabled && (
            <p className="text-xs text-gray-500 italic">Toggle to filter by delivery time</p>
          )}
        </div>

        {/* Categories */}
        <div className="lg:mb-6">
          <h4 className="font-medium mb-3 text-sm lg:text-base">Categories</h4>
          <div className="grid grid-cols-2 lg:grid-cols-1 gap-2">
            {categories.map((category) => (
              <label key={category} className="flex items-center text-sm cursor-pointer group transition-all duration-200 hover:bg-gray-50 p-1 rounded">
                <input
                  type="checkbox"
                  checked={filters.categories.includes(category)}
                  onChange={(e) => handleCategoryChange(category, e.target.checked)}
                  className="mr-2 w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 rounded focus:ring-blue-500 focus:ring-2 transition-all duration-200"
                />
                <span className="text-xs lg:text-sm group-hover:text-gray-900 transition-colors duration-200">{category}</span>
              </label>
            ))}
          </div>
        </div>

        {/* Brands */}
        <div className="lg:mb-6">
          <h4 className="font-medium mb-3 text-sm lg:text-base">Brand</h4>
          <div className="grid grid-cols-2 lg:grid-cols-1 gap-2">
            {brands.map((brand) => (
              <label key={brand} className="flex items-center text-sm cursor-pointer group transition-all duration-200 hover:bg-gray-50 p-1 rounded">
                <input
                  type="checkbox"
                  checked={filters.brands.includes(brand)}
                  onChange={(e) => handleBrandChange(brand, e.target.checked)}
                  className="mr-2 w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 rounded focus:ring-blue-500 focus:ring-2 transition-all duration-200"
                />
                <span className="text-xs lg:text-sm group-hover:text-gray-900 transition-colors duration-200">{brand}</span>
              </label>
            ))}
          </div>
        </div>

        {/* Rating */}
        <div className="lg:mb-6">
          <h4 className="font-medium mb-3 text-sm lg:text-base">Customer Rating</h4>
          <div className="grid grid-cols-2 lg:grid-cols-1 gap-2">
            {[4, 3, 2, 1].map((rating) => (
              <label key={rating} className="flex items-center text-sm cursor-pointer group transition-all duration-200 hover:bg-gray-50 p-1 rounded">
                <input
                  type="radio"
                  name="rating"
                  checked={filters.rating === rating}
                  onChange={() => onChange({ ...filters, rating })}
                  className="mr-2 w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 focus:ring-blue-500 focus:ring-2 transition-all duration-200"
                />
                <span className="text-xs lg:text-sm group-hover:text-gray-900 transition-colors duration-200">{rating}★ & up</span>
              </label>
            ))}
          </div>
        </div>

        {/* Price Range */}
        <div className="lg:mb-6">
          <h4 className="font-medium mb-3 text-sm lg:text-base">Price</h4>
          <div className="flex space-x-2 mb-2">
            <input
              type="number"
              placeholder="Min"
              value={filters.priceMin || ""}
              onChange={(e) => onChange({ ...filters, priceMin: Number(e.target.value) || 0 })}
              className="w-full lg:w-20 px-2 py-1 border rounded text-sm transition-all duration-200 focus:border-blue-500 focus:ring-1 focus:ring-blue-500 hover:border-gray-400"
            />
            <span className="text-sm self-center">to</span>
            <input
              type="number"
              placeholder="Max"
              value={filters.priceMax || ""}
              onChange={(e) => onChange({ ...filters, priceMax: Number(e.target.value) || 50000 })}
              className="w-full lg:w-20 px-2 py-1 border rounded text-sm transition-all duration-200 focus:border-blue-500 focus:ring-1 focus:ring-blue-500 hover:border-gray-400"
            />
          </div>
        </div>
      </div>
    </div>
  )
}
